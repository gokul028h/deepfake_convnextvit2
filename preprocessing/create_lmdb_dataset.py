import os
import cv2
import lmdb
import pickle
from tqdm import tqdm

DATASET_DIR = "../deep_fakes_explain/dataset"
OUTPUT_DIR = "../deep_fakes_explain/dataset"

TRAIN_DIR = os.path.join(DATASET_DIR, "training_set")
VAL_DIR = os.path.join(DATASET_DIR, "validation_set")

def collect_images(folder):

    data = []

    for label, cls in [(0,"real"),(1,"fake")]:

        path = os.path.join(folder, cls)

        for img in os.listdir(path):

            if img.endswith(".jpg") or img.endswith(".png"):

                data.append((os.path.join(path,img), label))

    return data


def create_lmdb(data, save_path):

    env = lmdb.open(save_path, map_size=1099511627776)

    with env.begin(write=True) as txn:

        for i,(img_path,label) in enumerate(tqdm(data)):

            img = cv2.imread(img_path)

            key = f"{i:08}".encode()

            value = pickle.dumps((img,label))

            txn.put(key,value)

        txn.put(b'length', str(len(data)).encode())

    env.close()


if __name__ == "__main__":

    train_data = collect_images(TRAIN_DIR)
    val_data = collect_images(VAL_DIR)

    print("Train images:",len(train_data))
    print("Val images:",len(val_data))

    create_lmdb(train_data, os.path.join(OUTPUT_DIR,"train_lmdb"))
    create_lmdb(val_data, os.path.join(OUTPUT_DIR,"val_lmdb"))