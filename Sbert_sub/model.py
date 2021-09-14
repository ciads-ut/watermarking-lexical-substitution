from pytorch_lightning import LightningModule
import torch.nn.functional as F
import torch
import math
from transformers import AutoModel, MPNetForMaskedLM, MPNetModel
from swords_epoch import swords_gen

import os
from util import *
import hparams

torch.autograd.set_detect_anomaly(True)


class Discriminator(LightningModule):
    '''
    Standard discriminator, but based on text
    self.bert: standard transformer model with output logits
    self.head: linear discriminator classifier head

    forward:
        input:
            attention mask: standard transformer attention mask
            input_one_hot: the input sentence as a one hot vector. converted this way because the generator outputs
                a one hot vector
            masked_token_pos: the position of the changed token / intended target. The discriminator head is dynamically
                attached to this position through the gather command
        output:
            a float in [0,1], whether the input is real (1) or fake (0)
    '''
    def __init__(self):
        super().__init__()
        self.bert = MPNetModel.from_pretrained(os.path.join(os.getcwd(), 'mpnet-base'), num_labels=1)
        self.head = torch.nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, attention_mask, input_one_hot, masked_token_pos):
        embeddings = torch.matmul(input_one_hot, self.bert.get_input_embeddings().weight)
        bert_output = self.bert(inputs_embeds=embeddings, attention_mask=attention_mask).last_hidden_state
        mask_output = bert_output.gather(1, (
                masked_token_pos.view(-1, 1) + torch.zeros((1, 1, bert_output.shape[2]), dtype=torch.int).type_as(
            masked_token_pos)).reshape(-1, 1, bert_output.shape[2]))
        return torch.sigmoid(self.head(mask_output))


class Generator(LightningModule):
    '''
    Standard generator, but based on text
    self.bert: the pretrained model which outputs logits for every word

    forward:
        input:
            attention mask: standard transformer attention mask
            input_ids: the ids of the input sentence, as received as the tokenizer. unused0 tokens surround the target
                word to provide the generator position information about the word it generates a synonym for
            masked_token_one_hot: the one hot representation of the target token. Equivalent to the one hot value of
                input ids at masked_token_pos. Multiplied by infinity and subtracted from logits to prevent copying
                of the target, the default output for bert
            masked_token_pos: the position of the target token in the sentence. Used to direct selecting the logits
                from the target's position.
            out_a: Kept for compatibility with lex_sub. Always 0 in sbert_sub
            softmax: if True, output is a softmax distribution. Used for test time to output probabilities. If False,
                output is gumbel_softmax, a one-hot vector to feed into the discriminator capable of backprop

    '''
    def __init__(self):
        super().__init__()
        self.bert = MPNetForMaskedLM.from_pretrained(os.path.join(os.getcwd(), 'mpnet-base'))

    def forward(self, attention_mask, input_ids, masked_token_one_hot, masked_token_pos, out_a=0, softmax=False):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).logits
        mask_output = bert_output.gather(1, (
                masked_token_pos.view(-1, 1) + torch.zeros((1, 1, bert_output.shape[2]), dtype=torch.int).type_as(
            masked_token_pos)).reshape(-1, 1, bert_output.shape[2]))
        if softmax:
            return F.softmax(mask_output * (1 - (masked_token_one_hot + out_a)) - (masked_token_one_hot + out_a) * 1e10,
                dim=-1)
        return F.gumbel_softmax(
            F.log_softmax(mask_output * (1 - (masked_token_one_hot + out_a)) - (masked_token_one_hot + out_a) * 1e10,
                dim=-1), tau=hparams.GUMBEL_TAU, hard=True)


class Sbert(LightningModule):
    '''
    Lightning wrapper module for Sbert model to allow freeze. This is a save of the current state of the art stsb model
    '''
    def __init__(self):
        super().__init__()
        self.Sbert = AutoModel.from_pretrained("stsb-mpnet-base-v2")

    def forward(self, **kwargs):
        return self.Sbert(**kwargs)


class SbertModel(LightningModule):
    def __init__(self, alpha):
        super().__init__()
        self.Disc = Discriminator()
        if hparams.TYPE == 'normal':
            self.Disc_B = Discriminator()
        elif hparams.TYPE == 'ema':
            self.Disc_B = EMA(self.Disc, hparams.DECAY)
        self.Gen_A = Generator()
        self.Sbert = Sbert()
        self.Sbert.freeze()
        self.alpha = alpha
        self.epochs = 0
        self.save_hyperparameters()

    '''
    How the model is trained
    input:
        attention_mask: standard attention mask
        original_sent_ids: sentence ids with no change. Plain english text
        padded_sent_ids: sentence ids with unused0 tokens around the target word to provide location information.
            Untested to see whether this information helps or harms
        masked_token_id:
            the id of the target word
        masked_token_pos: the position in the original sentence of the target word. Adding one gives the position in the
            padded sentence
    
    procedure:
        pass the padded sentence ids through the generator to receive a substitute token
        if a generator step: train the generator to minimize discriminator accuracy and sbert distance. Loss 
            relationship given by alpha
        if a discriminator step: train the discriminator to correctly identify generated tokens
            - previous versions of this worked with two discriminators. Toggled by setting TYPE
    '''

    def training_step(self, batch, batch_idx, optimizer_idx):
        attention_mask, original_sent_ids, padded_sent_ids, masked_token_id, masked_token_pos = batch
        masked_token_one_hot = F.one_hot(masked_token_id, tokenizer.vocab_size).type_as(masked_token_id)
        attention_mask_gen = torch.cat([torch.ones(attention_mask.shape[0], 2).type_as(attention_mask), attention_mask],
            dim=-1)

        out_gen = self.Gen_A(attention_mask_gen, padded_sent_ids, masked_token_one_hot, masked_token_pos + 1)
        original_sent_one_hot = F.one_hot(original_sent_ids, tokenizer.vocab_size).type_as(out_gen)
        gen_sent = original_sent_one_hot.clone()
        gen_sent[torch.arange(gen_sent.shape[0]), masked_token_pos.view(-1), :] = out_gen.reshape(-1,
            tokenizer.vocab_size)

        if optimizer_idx == 1:
            a_logits = self.Disc(attention_mask, gen_sent, masked_token_pos)
            gen_loss = F.binary_cross_entropy(a_logits, torch.ones(a_logits.shape).type_as(a_logits)) / math.log(2)
            self.log('gen_a', gen_loss, prog_bar=True, logger=True)
            sbert_orig = self.mean_pooling(self.Sbert(input_ids=original_sent_ids, attention_mask=attention_mask),
                attention_mask)
            gen_embeds = torch.matmul(gen_sent, self.Sbert.Sbert.get_input_embeddings().weight)
            sbert_gen = self.mean_pooling(self.Sbert(inputs_embeds=gen_embeds, attention_mask=attention_mask),
                attention_mask)
            sbert_loss = (1 - torch.mean(F.cosine_similarity(sbert_orig, sbert_gen))) * self.alpha
            self.log('sbert', sbert_loss, prog_bar=True, logger=True)

            gen_b_loss = 0
            if hparams.TYPE != 'single':
                b_logits = self.Disc_B(attention_mask, gen_sent, masked_token_pos)
                gen_b_loss = F.binary_cross_entropy(b_logits, torch.ones(b_logits.shape).type_as(b_logits)) / math.log(2)
                self.log('gen_b', gen_b_loss, prog_bar=True, logger=True)

            return gen_loss + sbert_loss + gen_b_loss

        elif optimizer_idx == 0:  # Let 1 be real, 0 fake
            if hparams.TYPE == 'ema':
                self.Disc_B.update()
            disc_gen_logits = self.Disc(attention_mask, gen_sent.detach(), masked_token_pos)
            disc_gen_loss = F.binary_cross_entropy(disc_gen_logits,
                torch.zeros(disc_gen_logits.shape).type_as(disc_gen_logits)) / math.log(2)
            self.log('disc_gen', disc_gen_loss, prog_bar=True, logger=True)

            disc_orig_logits = self.Disc(attention_mask, original_sent_one_hot, masked_token_pos)
            disc_orig_loss = F.binary_cross_entropy(disc_orig_logits,
                torch.ones(disc_orig_logits.shape).type_as(disc_orig_logits)) / math.log(2)
            self.log('disc_orig', disc_orig_loss, prog_bar=True, logger=True)
            return disc_gen_loss + disc_orig_loss

        else:
            disc_b_gen_logits = self.Disc_B(attention_mask, gen_sent.detach(), masked_token_pos)
            disc_b_gen_loss = F.binary_cross_entropy(disc_b_gen_logits,
                torch.zeros(disc_b_gen_logits.shape).type_as(disc_b_gen_logits)) / math.log(2)
            self.log('disc_b_gen', disc_b_gen_loss, prog_bar=True, logger=True)
            disc_b_orig_logits = self.Disc_B(attention_mask, original_sent_one_hot, masked_token_pos)
            disc_b_orig_loss = F.binary_cross_entropy(disc_b_orig_logits,
                torch.ones(disc_b_orig_logits.shape).type_as(disc_b_orig_logits)) / math.log(2)
            self.log('disc_b_orig', disc_b_orig_loss, prog_bar=True, logger=True)

            return disc_b_gen_loss + disc_b_orig_loss

    def configure_optimizers(self):
        Disc_opt = torch.optim.Adam([
            {"params": self.Disc.parameters()},
        ], lr= hparams.DISC_LEARNING_RATE)
        if hparams.TYPE == 'normal':
            Disc_B_opt = torch.optim.Adam([
                {"params": self.Disc_B.parameters()},
            ], lr=hparams.DISC_LEARNING_RATE)
        Gen_A_opt = torch.optim.Adam([
            {"params": self.Gen_A.parameters()},
        ], lr=hparams.GEN_LEARNING_RATE)
        if hparams.TYPE == 'normal':
            return [
                {'optimizer': Disc_opt},
                {'optimizer': Gen_A_opt},
                {'optimizer': Disc_B_opt},
            ]
        elif hparams.TYPE in ['ema', 'single']:
            return [
                {'optimizer': Disc_opt},# "frequency": 1},
                {'optimizer': Gen_A_opt},# "frequency": 0},
            ]

    def training_epoch_end(self, batch):
        # Runs similar version of swords_gen.py using swords_epoch.py. Acts as a validation set.
        swords_gen(self, self.epochs)
        self.epochs += 1

    def mean_pooling(self, model_output, attention_mask):
        # Used to calculate final sbert embeddings
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def generate(self, attention_mask, padded_sent_ids, masked_token_id, masked_token_pos):
        # For swords
        masked_token_one_hot = F.one_hot(masked_token_id, tokenizer.vocab_size).type_as(masked_token_id)
        return self.Gen_A(attention_mask, padded_sent_ids, masked_token_one_hot, masked_token_pos + 1, softmax=True)



from copy import deepcopy
from collections import OrderedDict


class EMA(LightningModule):
    '''
    Adapted from https://www.zijianhu.com/post/pytorch/ema/
    Facilitates having an EMA with a decay parameter. Used at both train and test time if hparams.TYPE='ema'
    '''

    def __init__(self, model: LightningModule, decay: float):
        super().__init__()
        self.decay = decay

        self.model = model
        self.shadow = deepcopy(self.model)

        for param in self.shadow.parameters():
            param.detach_()

    @torch.no_grad()
    def update(self):

        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())

        assert model_params.keys() == shadow_params.keys()

        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            shadow_params[name].sub_((1. - self.decay) * (shadow_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            shadow_buffers[name].copy_(buffer)

    def forward(self, attention_mask, gen_sent, masked_token_pos):
        return self.shadow(attention_mask, gen_sent, masked_token_pos)  # We want the shadow while training as well
