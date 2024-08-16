from datasets import load_dataset
from PIL import Image
import io
import os
import json
import tqdm
from concurrent.futures import ProcessPoolExecutor

model_list = [
    "DALLE_3",
    "DeepFloyd_I_XL_v1",
    "Midjourney_6",
    "SDXL_2_1",
    "SDXL_Base",
    "SDXL_Turbo",
]

ds = load_dataset("BaiqiL/GenAI-Bench-1600")["train"]

output_folder = "genai_bench_ds"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(f"{output_folder}/images", exist_ok=True)
os.makedirs(f"{output_folder}/metadata", exist_ok=True)


def byte_to_pil(byte):
    return Image.open(io.BytesIO(byte))


def process_item(item_index):
    item = ds[item_index]
    for model in model_list:
        metadata = {}
        metadata["prompt"] = item["prompt"]
        image = item[model]
        pil_image = byte_to_pil(image)
        ranking = item[f"{model}_HumanRating"]
        metadata["ranking"] = ranking
        pil_image.save(f"{output_folder}/images/{item_index}_{model}.png")
        with open(f"{output_folder}/metadata/{item_index}_{model}.json", "w") as f:
            json.dump(metadata, f)
    return len(model_list)


# Initialize the progress bar
pbar = tqdm.tqdm(total=len(ds) * len(model_list))

# Use ProcessPoolExecutor for parallel processing
with ProcessPoolExecutor() as executor:
    for result in executor.map(process_item, range(len(ds))):
        pbar.update(result)

pbar.close()
