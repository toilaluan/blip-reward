import requests
from PIL import Image
from transformers import AutoProcessor
from florence.modeling_florence2 import Florence2ForConditionalGeneration
import torch.nn as nn
import torch


class FlorenceAestheticModel(nn.Module):
    def __init__(self, model_name="microsoft/Florence-2-base"):
        super(FlorenceAestheticModel, self).__init__()
        self.model = Florence2ForConditionalGeneration.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.language_model.encoder.layers[-1].parameters():
            param.requires_grad = True
        self.mlp = nn.Linear(self.model.language_model.config.d_model, 1)

    def forward(self, input_ids, attention_mask, pixel_values, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
            **kwargs
        )
        hidden_states = outputs.encoder_hidden_states[-1]
        # hidden_states = hidden_states.mean(dim=1)
        hidden_states = hidden_states[:, 0, :]
        aesthetic_score = self.mlp(hidden_states)
        return aesthetic_score


if __name__ == "__main__":

    model = FlorenceAestheticModel()
    model.to("cuda")

    from data.pair_dataset import PairDataset, collate_fn
    from torch.utils.data import DataLoader

    data_root = "dataset/synthetic_ds"
    val_dataset = PairDataset(data_root, mode="val", processor="florence")
    val_dataloader = DataLoader(
        val_dataset, batch_size=16, shuffle=True, num_workers=1, collate_fn=collate_fn
    )
    batch = next(iter(val_dataloader))
    aesthetic_score = model(**batch["img_better"].to("cuda"))

    print(aesthetic_score)
