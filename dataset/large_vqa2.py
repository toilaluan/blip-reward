from datasets import load_dataset
from tqdm import tqdm
import os
import json
import concurrent.futures
from itertools import repeat

def process_item(item, output_folder):
    try:
        if item['answer_type'] != 'yes/no':
            return
        question = item['question']
        answer = item['multiple_choice_answer']
        question_id = item['question_id']
        image = item['image']
        image_id = item['image_id']
        image_file = f"{output_folder}/images/{image_id}.jpg"
        image.save(image_file)
        metadata = {
            "image_path": f"images/{image_id}.jpg",
            "question": question,
            "answer": answer
        }
        meta_file = f"{output_folder}/metadatas/{question_id}.json"
        with open(meta_file, "w") as f:
            json.dump(metadata, f)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    ds = load_dataset("lmms-lab/VQAv2", split="validation", num_proc=4)
    output_folder = "large_vqa"
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(f"{output_folder}/images", exist_ok=True)
    os.makedirs(f"{output_folder}/metadatas", exist_ok=True)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_item, ds, repeat(output_folder)), total=len(ds)))
