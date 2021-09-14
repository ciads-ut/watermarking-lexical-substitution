from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningDataModule
from itertools import islice
import torch
import hparams
import os
import random
from util import tokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Fixes a warning

rng = random.Random(hparams.seed)

class LexDataModule(LightningDataModule):

    '''
        Loads data and returns attention_mask, masked_sent_ids, padded_sent_ids, masked_token_ids, masked_token_pos
        attention_mask: standard attention mask
        original_sent_ids: sentence ids with no change. Plain english text
        padded_sent_ids: sentence ids with unused0 tokens around the target word to provide location information.
            Untested to see whether this information helps or harms
        masked_token_id:
            the id of the target word
        masked_token_pos: the position in the original sentence of the target word. Adding one gives the position in the
            padded sentence
    '''

    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.data = {}
        self.batch_size = batch_size
    
    def rel_path(self, path):
        return os.path.join(self.data_dir, path)

    def prepare_data(self):
        if os.path.isfile(self.rel_path('train.txt')): return
        else:
            raise FileExistsError("Could not find train.txt")

    # To load unselected, low quality substitutes from wikipedia, use the following setup function

    # def setup(self, stage = None):
    #     if type(stage) is not str:
    #         stage = stage.value
    #     cases = hparams.CASES
    #     for stage in ([stage] if stage else ['fit', 'validate', 'test']):
    #         text_name = {
    #             'fit': (self.rel_path('train.txt')),
    #             'validate': (self.rel_path('valid.txt')),
    #             'test': (self.rel_path('test.txt')),
    #         }[stage]
    #         lines = [line.strip() for line in islice(open(text_name, 'r'), cases)]
    #         tokenized = tokenizer(lines, return_tensors='pt', padding=True, truncation=True, max_length=hparams.MAX_LENGTH)
    #
    #         masked_sent_ids = []
    #         padded_sent_ids = []
    #         masked_token_ids = []
    #         masked_token_pos = []
    #         for line in tokenized.input_ids:
    #             for i in range(5):
    #                 index = rng.randint(1, hparams.MAX_LENGTH - 2)  # Don't get special tokens, either
    #                 if tokenizer.decode(line[index])[0:2] != "##" and tokenizer.decode(line[index + 1])[0:2] != "##":
    #                     break
    #             masked_token_ids.append([int(line[index])])
    #             padded_sent_ids.append(torch.cat([line[:index], torch.tensor([1, line[index], 1]), line[index+1:]]).reshape(-1,hparams.MAX_LENGTH+2))
    #             line[index] = tokenizer.mask_token_id
    #             masked_sent_ids.append(line.reshape(-1,hparams.MAX_LENGTH))
    #             masked_token_pos.append([index])
    #
    #         masked_sent_ids = torch.cat(masked_sent_ids)
    #         padded_sent_ids = torch.cat(padded_sent_ids)
    #         masked_token_ids = torch.tensor(masked_token_ids)
    #         masked_token_pos = torch.tensor(masked_token_pos)
    #         self.data[stage] = (tokenized.attention_mask, masked_sent_ids, padded_sent_ids, masked_token_ids, masked_token_pos)


    # To load selected, solid quality substitutes from wikipedia, use the following setup function

    def setup(self, stage = None):
        if type(stage) is not str:
            stage = stage.value
        cases = hparams.CASES
        for stage in ([stage] if stage else ['fit', 'validate', 'test']):
            text_name = {
                'fit': (self.rel_path('train_syn.txt')),
                'validate': (self.rel_path('valid_syn.txt')),
                'test': (self.rel_path('test_syn.txt')),
            }[stage]


            masked_sent_ids = []
            padded_sent_ids = []
            masked_token_ids = []
            masked_token_pos = []
            attention_mask = []
            for line in islice(open(text_name, 'r'), cases):
                start = line.find(" ")
                masked_token = line[:start]
                masked_token_id = tokenizer.encode(masked_token, add_special_tokens=False, return_tensors='pt')[0]
                if len(masked_token_id) > 1:
                    continue
                tokenized = tokenizer(line[start+1:], return_tensors='pt', padding=True, pad_to_multiple_of=hparams.MAX_LENGTH, max_length=hparams.MAX_LENGTH, truncation=True)
                try:
                    index = int((tokenized.input_ids == tokenizer.mask_token_id)[0].nonzero())
                except:
                    continue
                attention_mask.append(tokenized.attention_mask)
                masked_sent_ids.append(tokenized.input_ids)
                masked_token_pos.append([index])
                masked_token_ids.append([int(masked_token_id[0])])
                padded_sent_ids.append(torch.cat([tokenized.input_ids[0,:index], torch.tensor([1, masked_token_id, 1]), tokenized.input_ids[0,index+1:]]).reshape(-1, hparams.MAX_LENGTH+2))


            attention_mask = torch.cat(attention_mask)
            masked_sent_ids = torch.cat(masked_sent_ids)
            padded_sent_ids = torch.cat(padded_sent_ids)
            masked_token_ids = torch.tensor(masked_token_ids)
            masked_token_pos = torch.tensor(masked_token_pos)
            self.data[stage] = (attention_mask, masked_sent_ids, padded_sent_ids, masked_token_ids, masked_token_pos)

    def dataloader(self, stage: str):
        attention_mask, masked_sent_ids, padded_sent_ids, masked_token_id, masked_token_pos = self.data[stage]
        dataset = TensorDataset(attention_mask, masked_sent_ids, padded_sent_ids, masked_token_id, masked_token_pos)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=hparams.DATA_WORKERS)

    def train_dataloader(self):
        return self.dataloader('fit')

    # def val_dataloader(self):
    #     return self.dataloader('validate')

    # def test_dataloader(self):
    #     return self.dataloader('test')

