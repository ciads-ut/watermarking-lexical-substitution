from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningDataModule
from itertools import islice
import torch
import hparams
import random
import os

"""
Loads data into watermarker
input_ids: Standard tokenizer input ids of a normal sentence input
attention_mask: Standard tokenizer attention mask
msg_compact: An explicit array of the bits. 1010 -> [1,0,1,0]
msg_one_hot: A one-hot array of the bits. Unused in this version. 0 -> [1,0], 1-> [0,1], 1010-> [0,1,1,0,0,1,1,0]
"""

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Fixes a warning

def encode_msgs(msg_str):
    msg_one_hot = torch.zeros(len(msg_str), 2 * hparams.MSG_LEN)
    msg_compact = torch.zeros(len(msg_str), hparams.MSG_LEN)
    for i in range(len(msg_str)):
        for j in range(hparams.MSG_LEN):
            msg_one_hot[i, 2*j + (1 if msg_str[i][j] == '1' else 0)] = 1
            msg_compact[i, j] = 1 if msg_str[i][j] == '1' else 0
    return msg_compact, msg_one_hot

class WMDataModule(LightningDataModule):

    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.data = {}
        self.batch_size = batch_size
    
    def rel_path(self, path):
        return os.path.join(self.data_dir, path)

    def prepare_data(self):
        if os.path.isfile(self.rel_path('train.txt')): return

        rng = random.Random(hparams.seed)
        total_len = 0
        lines = []
        with open(self.rel_path('combined.json'), 'r') as infile:
            for line in infile:
                lines.append(line)
                total_len += len(line) + 1
        # rng.shuffle(lines)

        threshold = total_len // 10
        test_len = 0
        test = []
        valid_len = 0
        valid = []
        train = []

        for i in range(len(lines)):
            if test_len < threshold:
                test.append(lines[i])
                test_len += len(lines[i]) + 1
            elif valid_len < threshold:
                valid.append(lines[i])
                valid_len += len(lines[i]) + 1
            else:
                train.append(lines[i])

        for text_filename, msg_filename, lines in [('test.txt', 'test.msg.txt', test),
            ('valid.txt', 'valid.msg.txt', valid),
            ('train.txt', 'train.msg.txt', train)]:

            with open(self.rel_path(text_filename), 'w') as textfile:
                with open(self.rel_path(msg_filename), 'w') as msgfile:
                    rng = random.Random(hparams.seed)
                    current_len = 0
                    for line in lines:
                        for sent in hparams.senter(line).sents:
                            sent = sent.text.strip()
                            after_len = current_len + 1 + len(sent)
                            if abs(after_len - hparams.CHAR_TARGET) < \
                                    abs(current_len - hparams.CHAR_TARGET) or \
                                    current_len < 3 * hparams.CHAR_TARGET // 4:

                                next_sent = None
                                if after_len > 5 * hparams.CHAR_TARGET // 4:
                                    try:
                                        space_index = sent.rindex(' ', 0, 5 * hparams.CHAR_TARGET // 4 - current_len)
                                    except ValueError:
                                        space_index = len(sent)
                                    next_sent = sent[space_index:].strip()
                                    sent = sent[:space_index].strip()

                                if current_len != 0:
                                    textfile.write(' ')
                                    current_len += 1
                                current_len += len(sent)
                                textfile.write(sent)
                                sent = next_sent

                            while sent:
                                for _ in range(hparams.MSG_LEN):
                                    msgfile.write(str(rng.randint(0, 1)))
                                msgfile.write('\n')
                                try:
                                    space_index = sent.index(' ', hparams.CHAR_TARGET)
                                except ValueError:
                                    space_index = len(sent)
                                sent = sent[:space_index].strip()
                                next_sent = sent[space_index:].strip()
                                current_len = len(sent)
                                textfile.write('\n')
                                textfile.write(sent)
                                sent = next_sent

                    for _ in range(hparams.MSG_LEN):
                        msgfile.write(str(rng.randint(0, 1)))
                    msgfile.write('\n')
                    textfile.write('\n')

    def setup(self, stage = None):
        if type(stage) is not str:
            stage = stage.value
        cases = hparams.CASES
        for stage in ([stage] if stage else ['fit', 'validate', 'test']):
            text_name, msg_name = {
                'fit': (self.rel_path('train.txt'), self.rel_path('train.msg.txt')),
                'validate': (self.rel_path('valid.txt'), self.rel_path('valid.msg.txt')),
                'test': (self.rel_path('test.txt'), self.rel_path('test.msg.txt')),
            }[stage]
            lines = [line.strip() for line in islice(open(text_name, 'r'), cases)]
            msgs = [line.strip() for line in islice(open(msg_name, 'r'), cases)]
            tokenized = hparams.tokenizer(lines, return_tensors='pt', padding=True, truncation=True, max_length=hparams.MAX_LENGTH, add_special_tokens=False)  # TODO make padding work
            self.data[stage] = {
                'sents': (tokenized.attention_mask, tokenized.input_ids),
                'msgs': encode_msgs(msgs)
            }

    def dataloader(self, stage: str):
        input_ids, attention_mask = self.data[stage]['sents']
        msg_compact, msg_one_hot = self.data[stage]['msgs']
        dataset = TensorDataset(input_ids, attention_mask, msg_compact, msg_one_hot)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=hparams.DATA_WORKERS)

    def train_dataloader(self):
        return self.dataloader('fit')

    # def val_dataloader(self):
    #     return self.dataloader('validate')

    # def test_dataloader(self):
    #     return self.dataloader('test')
