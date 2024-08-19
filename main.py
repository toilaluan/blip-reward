from data.binary_vqa_dataset import BinaryVQADataset
from torch.utils.data import DataLoader
from pl_main import LitMain
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import argparse
from transformers import AutoProcessor
from PIL import Image


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


data_root = "dataset/BinaryVQA2"

train_dataset = BinaryVQADataset(data_root, mode="train")
print(train_dataset[0])
val_dataset = BinaryVQADataset(data_root, mode="val")

processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-base", trust_remote_code=True
)


def collate_fn(batch):
    questions = [item["question"] for item in batch]
    answers = [str(item["answer"]) for item in batch]
    image_files = [Image.open(item["image_file"]).convert("RGB") for item in batch]
    inputs = processor(
        text=questions,
        images=image_files,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    labels = processor.tokenizer(
        text=answers,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_token_type_ids=False,
    )
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

lit_main = LitMain()

wandb_logger.watch(lit_main, log_freq=500)


# trainer = Trainer(
#     accelerator="gpu",
#     accumulate_grad_batches=1,
#     check_val_every_n_epoch=1,
#     precision="bf16",
#     callbacks=callbacks,
#     max_epochs=20,
#     val_check_interval=0.25,
#     log_every_n_steps=10,
#     logger=wandb_logger,
#     strategy="ddp",
# )

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
