from torch.utils.data import Dataset
import os
import glob
import json
import pandas as pd
import glob
from transformers import Blip2Processor
from PIL import Image
from tqdm import tqdm


class PairDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        json_files = glob.glob(os.path.join(root_dir, "metadatas/*.json"))
        json_datas = []
        for json_file in json_files:
            with open(json_file, "r") as f:
                json_datas.append(json.load(f))
        self.data_df = pd.DataFrame(json_datas)
        self.unique_prompts = self.data_df["prompt"].unique()
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        self.data = self.make_data()


    def __len__(self):
        return len(self.data)
    
    def make_data(self):
        data = []
        print("Unique prompts: ", len(self.unique_prompts))
        pbar = tqdm(total=len(self.unique_prompts), desc="Making data")
        for prompt in self.unique_prompts:
            img_set = []
            item_data = {}
            items = self.data_df[self.data_df["prompt"] == prompt]
            inputs = self.processor(text=prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=150)
            for i in range(len(items)):
                img_path = os.path.join(self.root_dir, items.iloc[i]["image_path"])
                pil_image = Image.open(img_path)
                pil_image = pil_image.convert("RGB")
                image = self.processor(images=pil_image, return_tensors="pt")
                img_set.append(image)
            
            labels = items["rank"].tolist()
            
            for id_l in range(len(labels)):
                for id_r in range(id_l+1, len(labels)):
                    if labels[id_l] < labels[id_r]:
                        item_data['img_better'] = img_set[id_l]
                        item_data['img_worse'] = img_set[id_r]
                    elif labels[id_l] > labels[id_r]:
                        item_data['img_better'] = img_set[id_r]
                        item_data['img_worse'] = img_set[id_l]
                    else:
                        continue
                
                    item_data['img_better'].update(inputs)
                    item_data['img_worse'].update(inputs)
                    data.append(item_data)
            pbar.update(1)

        return data

    def __getitem__(self, index):
        item = self.data[index]
        item["img_better"]["pixel_values"] = item["img_better"]["pixel_values"].squeeze(0)
        item["img_worse"]["pixel_values"] = item["img_worse"]["pixel_values"].squeeze(0)
        item["img_better"]["input_ids"] = item["img_better"]["input_ids"].squeeze(0)
        item["img_worse"]["input_ids"] = item["img_worse"]["input_ids"].squeeze(0)
        item["img_better"]["attention_mask"] = item["img_better"]["attention_mask"].squeeze(0)
        item["img_worse"]["attention_mask"] = item["img_worse"]["attention_mask"].squeeze(0)
        return item

    
if __name__ == "__main__":
    dataset = PairDataset("/workspace/t5-reward/dataset/validation")
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for i, data in enumerate(dataloader):
        print(data)
        print(type(data))
        print(data["img_better"]["input_ids"].shape)
        print(data["img_better"]["pixel_values"].shape)
        break
