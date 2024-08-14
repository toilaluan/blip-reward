import pytorch_lightning as L
import torch
from torch import nn
import torch.nn.functional as F
from model import QformerAesthetic
from peft import LoraConfig, TaskType
import schedulefree

class LitMain(L.LightningModule):
    def __init__(self, model: QformerAesthetic):
        super().__init__()
        peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
        self.model = model

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.LongTensor, attention_mask: torch.LongTensor=None, **kwargs):
        return self.model(pixel_values, input_ids, attention_mask)

    def loss_fn(self, better_score, worse_score):
        output = torch.cat([better_score, worse_score], dim=1)
        target = torch.zeros(output.size(0), dtype=torch.long).to(output.device)
        return F.cross_entropy(output, target)

    def training_step(self, batch, batch_idx):
        better_inputs = batch["img_better"]
        worse_inputs = batch["img_worse"]
        better_score = self.model(**better_inputs)
        worse_score = self.model(**worse_inputs)
        loss = self.loss_fn(better_score, worse_score)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        better_inputs = batch["img_better"]
        worse_inputs = batch["img_worse"]
        better_score = self.model(**better_inputs)
        worse_score = self.model(**worse_inputs)
        loss = self.loss_fn(better_score, worse_score)
        distances = better_score - worse_score
        accuracy = torch.sum(distances > 0).item() / distances.size(0)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_accuracy", accuracy, prog_bar=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        return super().on_validation_epoch_end()

    def configure_optimizers(self):
        optimizer = schedulefree.AdamWScheduleFree(self.parameters(), lr=1e-4)
        return optimizer
    
        