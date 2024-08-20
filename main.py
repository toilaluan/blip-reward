from data.binary_vqa_dataset import BinaryVQADataset
from data.large_vqa import LargeVQADataset
from torch.utils.data import DataLoader
from pl_main import LitMain
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import argparse
from transformers import AutoProcessor
from PIL import Image
import torch

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to run training with various options."
    )

    parser.add_argument("--project", default="t5-reward", help="Wandb project name")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=20, help="Max epochs")
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=0.25,
        help="Validation check interval",
    )
    parser.add_argument(
        "--log_every_n_steps", type=int, default=10, help="Log every n steps"
    )
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--precision", default="bf16", help="Precision")
    parser.add_argument("--accelerator", default="gpu", help="Accelerator")
    parser.add_argument(
        "--accumulate_grad_batches", type=int, default=1, help="Accumulate grad batches"
    )
    parser.add_argument("--strategy", default="ddp", help="Strategy")
    parser.add_argument("--devices", default=1)

    return parser.parse_args()


wandb_logger = WandbLogger(project="t5-reward")


class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")


callbacks = [
    MyPrintingCallback(),
    ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=3,
    ),
    LearningRateMonitor("step"),
]

args = parse_args()

train_dataset = LargeVQADataset("dataset/large_vqa", mode="train")


val_dataset = BinaryVQADataset("dataset/BinaryVQA2", mode="val")

print(train_dataset[0])
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-base", trust_remote_code=True, revision='refs/pr/6'
)

lit_main = LitMain()

config = lit_main.model.language_model.config

def collate_fn(batch):
    questions = [item["question"] for item in batch]
    labels = []
    for item in batch:
        if item["answer"]:
            labels.append(1)
        else:
            labels.append(0)
    labels = torch.Tensor(labels)
    image_files = [Image.open(item["image_file"]).convert("RGB") for item in batch]
    inputs = processor(
        text=questions,
        images=image_files,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    inputs["decoder_input_ids"] = torch.Tensor([[config.pad_token_id, config.decoder_start_token_id]]*len(questions)).long()
    
    return inputs, labels


train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=8,
    collate_fn=collate_fn,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=8,
    collate_fn=collate_fn,
)
print(next(iter(train_dataloader)))
print(len(train_dataloader), len(val_dataloader))


wandb_logger.watch(lit_main, log_freq=500)

trainer = Trainer(
    accelerator=args.accelerator,
    accumulate_grad_batches=args.accumulate_grad_batches,
    precision=args.precision,
    callbacks=callbacks,
    max_epochs=args.max_epochs,
    val_check_interval=args.val_check_interval,
    log_every_n_steps=args.log_every_n_steps,
    logger=wandb_logger,
    strategy=args.strategy,
    devices=args.devices,
)

trainer.fit(lit_main, train_dataloader, val_dataloader)
