from torch.utils.data import Dataset
import torch
import pandas as pd
import glob
import random
import json
from PIL import Image
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Define a function to load a single JSON file
def load_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

class LargeVQADataset(Dataset):
    def __init__(self, root_folder, mode="train", seed=42):
        random.seed(seed)
        self.root_folder = root_folder
        json_files = glob.glob(root_folder + "/metadatas/*.json")
        print(len(json_files))
        if mode == "train":
            json_files = json_files[:int(len(json_files)*0.9)]
        elif mode == "val":
            json_files = json_files[int(len(json_files)*0.9):]
        with ProcessPoolExecutor() as executor:
            json_data = list(tqdm(executor.map(load_json_file, json_files), total=len(json_files)))
        print(len(json_data))
        self.json_data = json_data
            
    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        item = self.json_data[idx]
        image = self.root_folder + "/" + item["image_path"]
        return {
            "question": item["question"],
            "answer": 1 if "yes" in item["answer"].lower() else 0,
            "image_file": image
        }
        


if __name__ == "__main__":
    dataset = LargeVQADataset("/workspace/blip-reward/dataset/large_vqa", mode="train")
    print(len(dataset))
    print(dataset[0])
    tp = 0
    fp = 0
    for i in range(len(dataset)):
        item = dataset[i]
        answer = item["answer"]
        if answer == 1:
            tp += 1
        else:
            fp += 1

    print(tp, fp)
