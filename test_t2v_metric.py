import json
import pandas as pd
import glob
import t2v_metrics

print("VQAScore models:")
t2v_metrics.list_all_vqascore_models()

clip_flant5_score = t2v_metrics.VQAScore(model="clip-flant5-xl")


json_files = glob.glob("dataset/synthetic_ds/metadatas/*.json")

json_datas = []
for json_file in json_files:
    with open(json_file, "r") as f:
        json_datas.append(json.load(f))


data_df = pd.DataFrame(json_datas)

unique_prompts = data_df["prompt"].unique()

for prompt in unique_prompts:
    items = data_df[data_df["prompt"] == prompt]
    if len(items) < 2:
        continue
    labels = items["rank"].tolist()
    img_paths = items["image_path"].tolist()

    # sort img_paths based on labels
    img_paths = [x for _, x in sorted(zip(labels, img_paths))]
    labels = sorted(labels)

    dataset = [
        {"images": img_paths, "texts": [prompt]},
    ]

    scores = clip_flant5_score(dataset)
    print(labels)
    print(scores)
    raise
