from diffusers import DiffusionPipeline
import glob
import json
import torch
from PIL import Image
import tqdm
import diffusers

category = {
    # "0": [
    #     {
    #         "model_id": "runwayml/stable-diffusion-v1-5",
    #         "params": {"width": 512, "height": 512, "num_inference_steps": 30},
    #         "uid": "sd15",
    #     },
    # ],
    "1": [
        {
            "model_id": "SG161222/RealVisXL_V4.0_Lightning",
            "params": {
                "num_inference_steps": 8,
                "width": 1024,
                "height": 1024,
                "negative_prompt": "worst quality, low quality",
                "guidance_scale": 1.0,
            },
            "uid": "sdxl",
            "scheduler": diffusers.DPMSolverSDEScheduler,
        },
    ],
}

mj_metadatas = glob.glob("synthetic_ds/metadatas/sd15_*.json")
mj_metadatas = [json.load(open(metadata)) for metadata in mj_metadatas]
prompts = [metadata["prompt"] for metadata in mj_metadatas]
print(f"Total prompts: {len(prompts)}")

total_model = 0
for model_configs in category.values():
    total_model += len(model_configs)

pbar = tqdm.tqdm(total=len(mj_metadatas) * total_model)

for rank, model_configs in category.items():
    for model_config in model_configs:
        model_id = model_config["model_id"]
        params = model_config["params"]
        uid = model_config["uid"]
        pipeline = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
        pipeline.scheduler = model_config.get("scheduler").from_config(
            pipeline.scheduler.config
        )
        pipeline.to("cuda")
        save_image_folder = f"synthetic_ds/images"
        for i, prompt in enumerate(prompts):
            image = pipeline(prompt=prompt, **params).images[0]
            image_file = f"{save_image_folder}/{uid}_{i}.png"
            image.save(image_file)
            metadata = {
                "prompt": prompt,
                "model": model_id,
                "rank": int(rank),
                "image_path": image_file,
            }
            with open(f"synthetic_ds/metadatas/{uid}_{i}.json", "w") as f:
                json.dump(metadata, f)

            pbar.update(1)
