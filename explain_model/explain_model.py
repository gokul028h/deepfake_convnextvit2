import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import AvgPool2d
import yaml
from PIL import Image
import os
from albumentations import Compose, PadIfNeeded
from baselines.EfficientViT.transforms.albu import IsotropicResize
from baselines.EfficientViT.evit_model10 import EfficientViT


############################################
# CONFIG PATHS
############################################

BASE_DIR = '../deep_fakes_explain/'
MODELS_PATH = os.path.join(BASE_DIR, "models")

EXAMPLES = [
    "examples/real.jpg",
    "examples/fake.jpg"
]

OUTPUT_DIR = "explanation"

CONFIG_PATH = "baselines/EfficientViT/explained_architecture.yaml"

MODEL_WEIGHTS = os.path.join(
    MODELS_PATH,
    "efficientnetB0_checkpoint72_All"
)


############################################
# ATTENTION FUNCTIONS
############################################

def avg_heads(cam, grad):

    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])

    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)

    return cam


def apply_self_attention_rules(R_ss, cam_ss):

    return torch.matmul(cam_ss, R_ss)



def generate_relevance(model, input):

    output = model(input, register_hook=True)

    model.zero_grad()

    output.backward(retain_graph=True)

    num_tokens = model.transformer.blocks[0].attn.get_attention_map().shape[-1]

    R = torch.eye(num_tokens).cuda()

    for blk in model.transformer.blocks:

        grad = blk.attn.get_attn_gradients()
        cam = blk.attn.get_attention_map()

        cam = avg_heads(cam, grad)

        R += apply_self_attention_rules(R, cam)

    return R[0, 1:]


############################################
# IMAGE PROCESSING
############################################


def create_base_transform(size):

    return Compose([
        IsotropicResize(
            max_side=size,
            interpolation_down=cv2.INTER_AREA,
            interpolation_up=cv2.INTER_CUBIC
        ),
        PadIfNeeded(
            min_height=size,
            min_width=size,
            border_mode=cv2.BORDER_CONSTANT
        ),
    ])


def show_cam_on_image(img, mask):

    heatmap = cv2.applyColorMap(
        np.uint8(255 * mask),
        cv2.COLORMAP_JET
    )

    heatmap = np.float32(heatmap) / 255

    cam = heatmap + np.float32(img)

    cam = cam / np.max(cam)

    return cam


############################################
# LOAD MODEL
############################################

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)


model = EfficientViT(
    config=config,
    channels=1280,
    selected_efficient_net=0
)


model.load_state_dict(
    torch.load(MODEL_WEIGHTS)
)

model.cuda()

model.eval()


############################################
# VISUALIZATION
############################################

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


down_sample = AvgPool2d(kernel_size=2)



def visualize(image_path):

    print("Processing:", image_path)

    image = Image.open(image_path)

    transform = create_base_transform(
        config['model']['image-size']
    )

    t_image = transform(
        image=cv2.imread(image_path)
    )['image']

    t_image = torch.tensor(
        np.transpose(t_image, (2, 0, 1))
    ).float()


    pred = torch.sigmoid(
        model(t_image.unsqueeze(0).cuda())
    )

    print("Prediction:", pred.item())


    relevance = generate_relevance(
        model,
        t_image.unsqueeze(0).cuda()
    )


    relevance = relevance.reshape(1,1,32,32)

    relevance = down_sample(relevance)

    relevance = torch.nn.functional.interpolate(
        relevance,
        scale_factor=14,
        mode='bilinear'
    )

    relevance = relevance.reshape(224,224).detach().cpu().numpy()


    relevance = (relevance - relevance.min()) / (
        relevance.max() - relevance.min()
    )


    img_np = t_image.permute(1,2,0).cpu().numpy()

    img_np = (img_np - img_np.min()) / (
        img_np.max() - img_np.min()
    )


    cam = show_cam_on_image(img_np, relevance)

    cam = np.uint8(255 * cam)


    save_path = os.path.join(
        OUTPUT_DIR,
        os.path.basename(image_path)
    )


    cv2.imwrite(save_path, cam)

    print("Saved:", save_path)



############################################
# RUN
############################################


for img in EXAMPLES:

    visualize(img)


print("\nDONE")