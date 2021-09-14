from model import MainModel
from pytorch_lightning import Trainer
from data import WMDataModule
import hparams

model = MainModel(hparams.LEARNING_RATE, hparams.LOSS_WEIGHTS, hparams.GUMBEL_TAU)
# model = MainModel.load_from_checkpoint("BART_Watermarker.ckpt")

summary_data = WMDataModule('data', batch_size=hparams.BATCH_SIZE) # Consistency: Input a hparams.py variable, or just take from hparams.py

trainer = Trainer(
    gpus=[1],
    accelerator='dp',
    auto_select_gpus=True,
    max_epochs=hparams.EPOCHS,
    checkpoint_callback=False,
    deterministic=True
)

trainer.fit(model, summary_data)
trainer.save_checkpoint("BART_Watermarker.ckpt")
