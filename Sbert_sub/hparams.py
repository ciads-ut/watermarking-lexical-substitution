import sys

# TYPE = 'normal'
# TYPE = 'ema'
TYPE = 'single'

ALPHA = 16
GUMBEL_TAU = .5
# Ema decay if ema is enabled to create a second discriminator. Unused in SOTA model.
DECAY = .99995

DISC_LEARNING_RATE = 2e-6
GEN_LEARNING_RATE = 2e-6

DATA_WORKERS = 8

BATCH_SIZE = 32
MAX_EPOCHS = 1
CASES = sys.maxsize

# Maximum Length of Input Text. Done to increase train and data creation speed.
# Increasing this value (and the data length) is low-hanging fruit to improve results.
MAX_LENGTH = 30

seed = 2
