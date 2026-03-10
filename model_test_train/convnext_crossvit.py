import torch
from torch import nn
import timm
from einops import rearrange


class ArtifactAttention(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels // 8, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(channels // 8, 1, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        attn = self.conv1(x)

        attn = self.relu(attn)

        attn = self.conv2(attn)

        attn = self.sigmoid(attn)

        return x * attn


class CrossAttention(nn.Module):

    def __init__(self, dim=768, heads=8):

        super().__init__()

        self.heads = heads
        self.scale = dim ** -0.5

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, return_attention=False):

        B, N, C = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = dots.softmax(dim=-1)

        out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.to_out(out)

        if return_attention:
            return out, attn

        return out


class ConvNeXtCrossViT(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = timm.create_model(
            'convnext_tiny',
            pretrained=True,
            num_classes=0,
            global_pool=''
        )
        self.backbone.stem[0] = torch.nn.Conv2d(6,96,kernel_size=4, stride=4,padding=0)  # Adjust first conv layer for 6-channel input

        self.artifact_attention = ArtifactAttention(768)

        self.cross_attn = CrossAttention(dim=768)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(768, 1)


    def forward(self, x):

        x = self.backbone.forward_features(x)

        x = self.artifact_attention(x)

        B, C, H, W = x.shape

        x = rearrange(x, 'b c h w -> b (h w) c')

        x = self.cross_attn(x)

        x = x.transpose(1, 2)

        x = self.pool(x).squeeze(-1)

        return self.fc(x)


    # Used for explainability (attention maps)
    def forward_attention(self, x):

        x = self.backbone.forward_features(x)

        x = self.artifact_attention(x)

        B, C, H, W = x.shape

        x = rearrange(x, 'b c h w -> b (h w) c')

        x, attn = self.cross_attn(x, return_attention=True)

        return x, attn
