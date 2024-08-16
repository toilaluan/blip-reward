from torch.utils.data import Dataset
import os
import glob
import json
import pandas as pd
import glob
from transformers import Blip2Processor
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

# processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-base", trust_remote_code=True
)


def collate_fn(batch):
    prompt, img_better, img_worse = zip(*batch)
    img_better = processor(
        images=img_better, return_tensors="pt", text=prompt, padding=True
    )
    img_worse = processor(
        images=img_worse, return_tensors="pt", text=prompt, padding=True
    )
    return {
        "img_better": img_better,
        "img_worse": img_worse,
    }


class PairDataset(Dataset):
    def __init__(self, root_dir, mode="train", processor="t5"):
        self.root_dir = root_dir
        json_files = glob.glob(os.path.join(root_dir, "metadatas/*.json"))
        print(len(json_files))
        json_datas = []
        for json_file in json_files:
            with open(json_file, "r") as f:
                json_datas.append(json.load(f))
        self.data_df = pd.DataFrame(json_datas)
        self.unique_prompts = self.data_df["prompt"].unique()
        if processor == "t5":
            self.processor = Blip2Processor.from_pretrained(
                "Salesforce/blip2-flan-t5-xl"
            )
        elif processor == "florence":
            self.processor = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-base", trust_remote_code=True
            )
        self.data = self.make_data()
        if mode == "train":
            self.data = self.data[: int(0.8 * len(self.data))]
        else:
            self.data = self.data[int(0.8 * len(self.data)) :]

    def __len__(self):
        return len(self.data)

    def make_data(self):
        data = []
        print("Unique prompts: ", len(self.unique_prompts))
        pbar = tqdm(total=len(self.unique_prompts), desc="Making data")
        for prompt in self.unique_prompts:
            item_data = {
                "prompt": f'Is this image describte "{prompt}"? How aesthetic it looks?'
            }
            items = self.data_df[self.data_df["prompt"] == prompt]
            if len(items) < 2:
                continue
            labels = items["rank"].tolist()
            print(labels)
            img_paths = items["image_path"].tolist()
            # inputs = self.processor(text=prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=150)
            # for i in range(len(items)):
            #     img_path = os.path.join(self.root_dir, items.iloc[i]["image_path"])
            #     pil_image = Image.open(img_path)
            #     pil_image = pil_image.convert("RGB")
            #     image = self.processor(images=pil_image, return_tensors="pt")
            #     img_set.append(image)

            for id_l in range(len(labels)):
                for id_r in range(id_l + 1, len(labels)):
                    if labels[id_l] > labels[id_r]:
                        item_data["img_better"] = img_paths[id_l]
                        item_data["img_worse"] = img_paths[id_r]
                    elif labels[id_l] < labels[id_r]:
                        item_data["img_better"] = img_paths[id_r]
                        item_data["img_worse"] = img_paths[id_l]
                    else:
                        continue

                    # item_data['img_better'].update(inputs)
                    # item_data['img_worse'].update(inputs)
                    data.append(item_data)
            pbar.update(1)

        return data

    def __getitem__(self, index):
        item = self.data[index]

        img_better_path = os.path.join(self.root_dir, item["img_better"]).replace(
            "synthetic_ds/synthetic_ds", "synthetic_ds"
        )
        img_worse_path = os.path.join(self.root_dir, item["img_worse"]).replace(
            "synthetic_ds/synthetic_ds", "synthetic_ds"
        )
        img_better = Image.open(img_better_path).convert("RGB")
        img_worse = Image.open(img_worse_path).convert("RGB")

        return item["prompt"], img_better, img_worse

    def visualize_sample(self, index):
        item = self.data[index]
        print(item["prompt"])
        img_better_path = os.path.join(self.root_dir, item["img_better"]).replace(
            "synthetic_ds/synthetic_ds", "synthetic_ds"
        )
        img_worse_path = os.path.join(self.root_dir, item["img_worse"]).replace(
            "synthetic_ds/synthetic_ds", "synthetic_ds"
        )
        print(img_better_path)
        print(img_worse_path)
        img_better = Image.open(img_better_path)
        img_worse = Image.open(img_worse_path)
        if img_better.size[0] != 1024:
            img_better = img_better.resize((1024, 1024))
        if img_worse.size[0] != 1024:
            img_worse = img_worse.resize((1024, 1024))
        # concatenate images
        concat_img = self._get_concat_h(img_better, img_worse)
        concat_img.save("concat_img.jpg")

    def _get_concat_h(self, im1, im2):
        dst = Image.new("RGB", (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst


if __name__ == "__main__":
    dataset = PairDataset("/workspace/blip-reward/dataset/synthetic_ds")
    print(len(dataset))
    import random

    dataset.visualize_sample(random.randint(0, len(dataset)))
    # from torch.utils.data import DataLoader

    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    # for i, data in enumerate(dataloader):
    #     print(data)
    #     print(type(data))
    #     print(data["img_better"]["input_ids"].shape)
    #     print(data["img_better"]["pixel_values"].shape)
    #     break
