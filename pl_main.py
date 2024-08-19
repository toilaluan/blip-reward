import pytorch_lightning as L
import torch
from torch import nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType
import schedulefree
from transformers import AutoProcessor, AutoModelForCausalLM


class LitMain(L.LightningModule):
    def __init__(
        self,
        pretrained_model_name="microsoft/Florence-2-base",
        torch_dtype=torch.float32,
    ):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, torch_dtype=torch_dtype, trust_remote_code=True
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor = None,
    ):
        return self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def loss_fn(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(**x).logits
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(**x).logits
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = schedulefree.AdamWScheduleFree(self.parameters(), lr=1e-4)
        return optimizer
