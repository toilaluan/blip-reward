from datasets import load_dataset
import datasets
import os
from PIL import Image
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_item(i, item, output):
    try:
        img_path = item['image']["path"]
        img = Image.open(img_path).convert("RGB")
        img.save(os.path.join(output, "images", f"{i}.png"))
        item.pop('image')
        item['image_path'] = os.path.join("images", f"{i}.png")
        item['version'] = 'legacy'
        with open(os.path.join(output, "metadatas", f"{i}.json"), "w") as f:
            json.dump(item, f)
    except Exception as e:
        print(f"Error: {e}")
    return i

def main():
    dataset = load_dataset("THUDM/ImageRewardDB", "8k")['validation']
    dataset = dataset.cast_column("image", datasets.Image(decode=False))

    output = "validation"
    os.makedirs(output, exist_ok=True)
    os.makedirs(os.path.join(output, "images"), exist_ok=True)
    os.makedirs(os.path.join(output, "metadatas"), exist_ok=True)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_item, i, item, output) for i, item in enumerate(dataset)]
        
        for future in as_completed(futures):
            i = future.result()
            if i % 100 == 0:
                print(f"Processed {i} images")

if __name__ == "__main__":
    main()
