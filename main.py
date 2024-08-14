from data.pair_dataset import PairDataset
from model import QformerAesthetic
from torch.utils.data import DataLoader
from pl_main import LitMain
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

wandb_logger = WandbLogger(project="t5-reward")


class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")
callbacks = [MyPrintingCallback(), ModelCheckpoint(monitor="val_loss", dirpath="/workspace/t5-reward/checkpoints", filename="best-checkpoint"), LearningRateMonitor("step")]

model = QformerAesthetic()


train_root = "dataset/train"
val_root = "dataset/validation"

train_dataset = PairDataset(train_root)
val_dataset = PairDataset(val_root)


train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=1)


lit_main = LitMain(model)

wandb_logger.watch(lit_main, log_freq=500)


trainer = Trainer(
    accelerator="gpu", 
    accumulate_grad_batches=1, 
    check_val_every_n_epoch=1, 
    precision="bf16", 
    callbacks=callbacks,
    max_epochs=20,
    val_check_interval=0.25,
    log_every_n_steps=10,
    logger=wandb_logger,
)


trainer.fit(lit_main, train_dataloader, val_dataloader)