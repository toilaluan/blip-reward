import diffusers
import os
import json
import datasets
import requests
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

N_SAMPLE_PER_MODEL = 1000

midjourney_v6_ds = datasets.load_dataset("CortexLM/midjourney-v6")["train"]

MJ_RANK = 5

output_folder = "synthetic_ds"
image_folder = "images"
metadata_folder = "metadatas"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(f"{output_folder}/images", exist_ok=True)
os.makedirs(f"{output_folder}/metadatas", exist_ok=True)


def get_and_cut_to_4(image_url):
    image = Image.open(requests.get(image_url, stream=True).raw)
    width, height = image.size

    # Image is a grid of 4 images
    image = image.crop((0, 0, width // 2, height // 2))
    return image


def process_sample(i, sample):
    try:
        image_save_path = f"{output_folder}/{image_folder}/mj_v6_{i}.png"
        image_url = sample["image_url"]
        image = get_and_cut_to_4(image_url)
        image.save(image_save_path)
        metadata = {
            "prompt": sample["prompt"],
            "model": "midjourney-v6",
            "rank": MJ_RANK,
            "image_path": f"{image_folder}/mj_v6_{i}.png",
        }
        with open(f"{output_folder}/{metadata_folder}/mj_v6_{i}.json", "w") as f:
            json.dump(metadata, f)
        return True
    except Exception as e:
        print(f"Error processing sample {i}: {e}")
        return False


def main():
    successes = 0
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_sample, i, sample)
            for i, sample in enumerate(midjourney_v6_ds)
        ]
        for future in tqdm(as_completed(futures), total=N_SAMPLE_PER_MODEL):
            if successes >= N_SAMPLE_PER_MODEL:
                break
            if future.result():
                successes += 1


if __name__ == "__main__":
    main()
