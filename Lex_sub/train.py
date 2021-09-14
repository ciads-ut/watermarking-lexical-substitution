from model import LexModel
from pytorch_lightning import Trainer
from data import LexDataModule
import hparams
from pytorch_lightning import loggers as pl_loggers

'''
How lex_sub is trained. The 2 Generator Model.
This model is trained with several steps to initialize the generator
1. Mutliply the padded_sent_ids input to generator a and generator b to create out_a and out_b by 0. This, in effect,
    generates random tokens, useful for helping the discriminator learn position information. Train for 100 steps with
    the frequency set to 1 for the discriminator and 0 for the two generators in configure optimizers.
2. Remove the multiplication on padded_sent_ids. Train for 3000 steps with the frequency set to 1 for the discriminator
    and 0 for the two generators in configure optimizers.
3. Remove the frequency attribute and train. This model is not currently functioning as intended and is in an early
    research state. Expect collapse to mask filling. For both generators. 
'''

model = LexModel()
# model = LexModel.load_from_checkpoint("lex_sub.ckpt")

summary_data = LexDataModule('data', batch_size=hparams.BATCH_SIZE)

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
trainer.save_checkpoint("lex_sub.ckpt")
