import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType
import schedulefree
from transformers import AutoProcessor, AutoModelForCausalLM
from torchmetrics import F1Score, Recall, Precision
class LitMain(L.LightningModule):
    def __init__(
        self,
        pretrained_model_name="toilaluan/Florence-2-large-binary-vqa",
        torch_dtype=torch.float32,
    ):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, torch_dtype=torch_dtype, trust_remote_code=True, ignore_mismatched_sizes=True
        )
        for param in self.model.parameters():
          param.requires_grad = False
        for param in self.model.language_model.model.decoder.layers[-3:].parameters():
            param.requires_grad = True
        for param in self.model.language_model.lm_head.parameters():
            param.requires_grad = True
        self.model.language_model.final_logits_bias.requires_grad = True
        self.recall_cal = Recall(task="binary")
        self.prec_cal = Precision(task="binary")
        self.f1_cal = F1Score(task="binary")

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor = None,
        labels: torch.LongTensor = None,
        decoder_input_ids: torch.LongTensor = None,
    ):
        return self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids
        )

    def loss_fn(self, logits, labels):
        return F.binary_cross_entropy_with_logits(logits, labels)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(**x)
        logits = output.logits[:,-1]
        logits = logits.squeeze(-1)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(**x)
        logits = output.logits[:,-1]
        logits = logits.squeeze(-1)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        f1 = self.f1_cal(logits, y)
        r = self.recall_cal(logits, y)
        p = self.recall_cal(logits, y)
        self.log("f1_val", f1, prog_bar=True, on_epoch=True, on_step=False)
        self.log("r_val", r, prog_bar=True, on_epoch=True, on_step=False)
        self.log("p_val", p, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        optimizer = schedulefree.AdamWScheduleFree(self.parameters(), lr=1e-4, weight_decay=1e-4)
        return optimizer
