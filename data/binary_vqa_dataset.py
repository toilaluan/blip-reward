from torch.utils.data import Dataset
import torch
import pandas as pd
import glob
import random


class BinaryVQADataset(Dataset):
    def __init__(self, root_folder, mode="train", seed=42):
        random.seed(seed)
        self.root_folder = root_folder
        self.image_files = (
            glob.glob(root_folder + "/images/*.png")
            + glob.glob(root_folder + "/images/*.jpg")
            + glob.glob(root_folder + "/images/*.jpeg")
        )
        self.id2image_file = {
            image_file.split("/")[-1].split(".")[0]: image_file
            for image_file in self.image_files
        }
        question_df = pd.read_csv(root_folder + "/questions.csv")
        if mode == "train":
            question_df = question_df.sample(frac=0.8, random_state=seed)
        elif mode == "val":
            question_df = question_df.drop(
                question_df.sample(frac=0.8, random_state=seed).index
            )
        self.questions = []

        for i, row in question_df.iterrows():
            row = list(row.dropna())
            image_id = row[0]
            image_id = "%04d" % image_id
            question_answers = row[1:]
            questions = []
            answers = []
            for q_a in question_answers:
                q, a = q_a.split("?")
                questions.append(q)
                b_a = 1 if "yes" in a.lower() else 0
                answers.append(b_a)
            for q, a in zip(questions, answers):
                self.questions.append(
                    {
                        "question": q,
                        "answer": a,
                        "image_file": self.id2image_file[image_id],
                    }
                )

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx]


if __name__ == "__main__":
    dataset = BinaryVQADataset("/workspace/blip-reward/dataset/BinaryVQA2", mode="val")
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
