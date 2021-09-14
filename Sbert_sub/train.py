from model import SbertModel
from pytorch_lightning import Trainer
from data import SbertDataModule
import hparams
from pytorch_lightning import loggers as pl_loggers

'''
How sbert_sub is trained.
For approximately 3000 steps, the discriminator is trained without the generator. This is because the generator is
initialized in a good position while the discriminator is not. Done by adjusting the frequency to 1, 0 in model.py
After this, the model trains as normal. Saving every epoch is suggested. On epoch end swords validation scores are saved
'''

model = SbertModel(alpha=hparams.ALPHA)
# model = SbertModel.load_from_checkpoint('sbert_sub.ckpt')
model.alpha = hparams.ALPHA

summary_data = SbertDataModule('data', batch_size=hparams.BATCH_SIZE)

tb_logger = pl_loggers.TensorBoardLogger('logs/')

trainer = Trainer(
    gpus=[1],
    accelerator='dp',
    auto_select_gpus=True,
    max_epochs=hparams.MAX_EPOCHS,
    deterministic=True,
    logger=tb_logger,
)

trainer.fit(model, summary_data)
# trainer.save_checkpoint("sbert_sub.ckpt")


