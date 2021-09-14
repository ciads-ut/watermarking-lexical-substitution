from transformers import BartTokenizerFast
import sys
import os

MSG_LEN = 4
LEARNING_RATE = 2e-5
HEAD_LEARNING_RATE = 2e-3
GUMBEL_TAU = 0.5
BIT_MULTIPLIER = 1
WEIGHT_DECAY = 0.01
LOSS_WEIGHTS = {'sbart': 50, 'bit_acc': 1, 'reconstruction': 0.2, 'discriminator': 1}

tokenizer = BartTokenizerFast.from_pretrained(os.path.join(os.getcwd(), 'bart-base'))
MAX_LENGTH = 20

seed = 1000

CHAR_TARGET = 5 * 80 * 2

USE_HARDWARE = True
DATA_WORKERS = 8

BATCH_SIZE = 32
EPOCHS = 15
CASES = sys.maxsize
