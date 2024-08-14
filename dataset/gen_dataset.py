from datasets import load_dataset
import os
from PIL import Image
import json

dataset = load_dataset("THUDM/ImageRewardDB", "4k")['train']

output = "train"
os.makedirs(output, exist_ok=True)
os.makedirs(os.path.join(output, "images"), exist_ok=True)
os.makedirs(os.path.join(output, "metadatas"), exist_ok=True)

for i, item in enumerate(dataset):
    img = item['image']
    img.save(os.path.join(output, "images", f"{i}.png"))
    item.pop('image')
    item['image_path'] = os.path.join("images", f"{i}.png")
    item['version'] = 'legacy'
    with open(os.path.join(output, "metadatas", f"{i}.json"), "w") as f:
        json.dump(item, f)
    if i % 100 == 0:
        print(f"Processed {i} images")