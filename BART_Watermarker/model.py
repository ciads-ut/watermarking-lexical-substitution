from pytorch_lightning import LightningModule
import torch.nn.functional as F
from itertools import chain
import torch.nn as nn
import torch
import math

from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from util import *
import hparams


class FrozenEncoder(LightningModule):
    """
    If Sbart and the Discriminator both are frozen, this class is used to make the same forward pass for both
    """
    def __init__(self):
        super().__init__()
        self.encoder = create_bart().get_encoder()
        self.head = nn.Linear(self.encoder.config.d_model, 1)

    def forward(self, attention_mask, input_ids=None, sent_one_hot=None):
        if sent_one_hot is not None:
            embeddings = torch.matmul(sent_one_hot, get_embeddings(self.encoder).weight)
        elif input_ids is not None:
            embeddings = get_embeddings(self.encoder)(input_ids)
        else:
            raise TypeError('Discriminator needs either sent_ids or sent_one_hot but got nothing')
        return self.encoder(inputs_embeds=embeddings, attention_mask=attention_mask)


class SbartModel(LightningModule):
    """
    Calculates sentence embeddings using a frozen bart encoder.
    """
    def __init__(self, frozen_encoder):
        super().__init__()
        self.encoder = frozen_encoder.encoder

    def forward(self, sent_mask, sent_ids, encoding):
        tgt_mean = self.encoder.forward(input_ids=sent_ids, attention_mask=sent_mask)[0]
        lm_mean = encoding[0]
        mse_loss_fct = torch.nn.MSELoss()
        return mse_loss_fct(tgt_mean, lm_mean)


class Discriminator(LightningModule):
    """
    Standard GAN Discriminator. Takes in a sentence and outputs whether it is watermarked or not
    """
    def __init__(self, use_frozen=False, frozen_encoder=None):
        super().__init__()
        self.use_frozen = use_frozen
        if self.use_frozen:
            self.encoder = frozen_encoder.encoder
        else:
            self.encoder = create_bart().get_encoder()
        self.head = nn.Linear(self.encoder.config.d_model, 1)

    def forward(self, sent_mask, sent_ids=None, sent_one_hot=None, encoder_output=None):
        if encoder_output is None:
            if sent_one_hot is not None:
                embeddings = torch.matmul(sent_one_hot, get_embeddings(self.encoder).weight)
            elif sent_ids is not None:
                embeddings = get_embeddings(self.encoder)(sent_ids)
            else:
                raise TypeError('Discriminator needs either sent_ids or sent_one_hot but got nothing')
            encoder_output = self.encoder(inputs_embeds=embeddings, attention_mask=sent_mask)
        pooled = torch.mean(encoder_output.last_hidden_state, dim=1)
        return (torch.sigmoid(self.head(pooled))).reshape(-1)


class MessageDecoder(LightningModule):
    """
    Applies a linear head with Message Length number of outputs to a BART encoder
    """
    def __init__(self):
        super().__init__()
        self.encoder = create_bart().get_encoder()
        self.head = nn.Linear(self.encoder.config.d_model, hparams.MSG_LEN)

    def forward(self, sent_mask, sent_ids=None, sent_one_hot=None, sent_embeds=None):
        if sent_embeds is None:
            if sent_one_hot is not None:
                sent_embeds = torch.matmul(sent_one_hot, get_embeddings(self.encoder).weight)
            elif sent_ids is not None:
                sent_embeds = get_embeddings(self.encoder)(sent_ids)
            else:
                raise TypeError('MessageDecoder needs either sent_ids, sent_one_hot, or sent_embeds but got nothing')
        encoder_output = self.encoder(inputs_embeds=sent_embeds, attention_mask=sent_mask)
        pooled = torch.mean(encoder_output.last_hidden_state, dim=1)
        return torch.sigmoid(self.head(pooled))


class WatermarkingModel(LightningModule):
    """
    Runs the input sentence through a BART encoder as well as the input bits through a linear layer.
    The BART encoder outputs are added to the linear outputs and fed into the decoder for output
    """
    def __init__(self):
        super().__init__()
        self.bart_model = create_bart()
        self.msg_encoder = nn.Linear(hparams.MSG_LEN, get_embedding_dims(self.bart_model))

    def forward(self, sent_mask, sent_ids, msg_orig):
        encoder_outputs = self.bart_model.model.encoder(attention_mask=sent_mask, input_ids=sent_ids).last_hidden_state
        msg_embeds = self.msg_encoder(msg_orig).reshape(-1, 1,
            get_embedding_dims(self.bart_model)) * hparams.BIT_MULTIPLIER
        add_output = BaseModelOutput(last_hidden_state=encoder_outputs + msg_embeds)
        return self.bart_model(encoder_outputs=add_output, labels=sent_ids)


class MainModel(LightningModule):
    """
    Connects all message components together in a standard GAN training loop. The generator receives additional
    losses from SBART and the Discriminator
    """
    def __init__(self, learning_rate, loss_weights, gumbel_tau):
        super().__init__()
        self.watermarker = WatermarkingModel()
        self.msg_decoder = MessageDecoder()
        self.frozen_encoder = FrozenEncoder()
        self.frozen_encoder.freeze()
        self.sbart_model = SbartModel(frozen_encoder=self.frozen_encoder)
        self.discriminator = Discriminator(use_frozen=False)
        self.learning_rate = learning_rate
        self.loss_weights = loss_weights
        self.gumbel_tau = gumbel_tau
        self.save_hyperparameters()

    def forward(self, sent_mask, sent_ids, msg_orig):
        wm_output = self.watermarker(sent_mask, sent_ids, msg_orig)
        wm_one_hot = F.gumbel_softmax(F.log_softmax(wm_output.logits, dim=-1),
            tau=self.gumbel_tau, hard=True)
        msg_output = self.msg_decoder(sent_mask, sent_one_hot=wm_one_hot)
        return wm_output, wm_one_hot, msg_output

    def configure_optimizers(self):
        optim_1 = torch.optim.Adam([
            {"params": self.watermarker.bart_model.parameters(), "lr": self.learning_rate},
            {"params": self.msg_decoder.parameters(), "lr": self.learning_rate},
            {"params": self.watermarker.msg_encoder.parameters(), "lr": hparams.HEAD_LEARNING_RATE,
                "weight_decay": hparams.WEIGHT_DECAY},
        ])
        optim_2 = torch.optim.Adam([
            {"params": self.discriminator.parameters(), "lr": self.learning_rate},
        ])
        return [optim_1, optim_2]

    def training_step(self, batch, batch_idx, optimizer_idx):
        sent_mask, sent_ids, msg_orig, msg_one_hot = batch
        wm_output, wm_one_hot, msg_output = self(sent_mask, sent_ids, msg_orig)
        msg_acc_loss = F.binary_cross_entropy(msg_output, msg_orig) / math.log(2)
        if optimizer_idx == 0:
            frozen_encoding = self.frozen_encoder(sent_mask, sent_one_hot=wm_one_hot)
            sbart_loss = self.sbart_model(sent_mask, sent_ids, frozen_encoding)
            discriminator_loss = self.discriminator(sent_mask, sent_one_hot=wm_one_hot)
            discriminator_loss = F.binary_cross_entropy(discriminator_loss,
                torch.ones(discriminator_loss.size(0)).type_as(discriminator_loss)) / math.log(2)
            self.log('bacc', msg_acc_loss * self.loss_weights['bit_acc'], prog_bar=True)
            self.log('rec', wm_output.loss * self.loss_weights['reconstruction'], prog_bar=True)
            self.log('disc_gen', discriminator_loss * self.loss_weights['discriminator'], prog_bar=True)
            self.log('sbart', sbart_loss * self.loss_weights['sbart'], prog_bar=True)
            self.log('sum', msg_acc_loss * self.loss_weights['bit_acc'] + \
                                 wm_output.loss * self.loss_weights['reconstruction'] + \
                                 discriminator_loss * self.loss_weights['discriminator'] + \
                                 sbart_loss * self.loss_weights['sbart'], prog_bar=True)
            return {'loss':
                msg_acc_loss * self.loss_weights['bit_acc'] + \
                wm_output.loss * self.loss_weights['reconstruction'] + \
                discriminator_loss * self.loss_weights['discriminator'] + \
                sbart_loss * self.loss_weights['sbart']}

        else:
            wm_one_hot = F.gumbel_softmax(F.log_softmax(wm_output.logits, dim=-1),
                tau=self.gumbel_tau, hard=True)
            wm_loss = self.discriminator(sent_mask, sent_one_hot=wm_one_hot.detach())
            wm_loss = F.binary_cross_entropy(wm_loss,
                torch.zeros(wm_loss.size(0)).type_as(wm_loss)) / math.log(2)
            orig_loss = self.discriminator(sent_mask, sent_ids=sent_ids.detach())
            orig_loss = F.binary_cross_entropy(orig_loss,
                torch.ones(orig_loss.size(0)).type_as(orig_loss)) / math.log(2)
            tqdm_dict = {'wm_loss': wm_loss, 'orig_loss': orig_loss}
            self.log('wm', wm_loss, prog_bar=True)
            self.log("orig", orig_loss, prog_bar=True)
            self.log("disc", wm_loss + orig_loss, prog_bar=True)
            return {'loss': wm_loss + orig_loss, 'progress_bar': tqdm_dict, 'log': tqdm_dict}
