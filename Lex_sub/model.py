from pytorch_lightning import LightningModule
import torch.nn.functional as F
import torch
import math
from swords_epoch import swords_gen

from util import *
import hparams

torch.autograd.set_detect_anomaly(True)


class Discriminator(LightningModule):
    """
    Non-Standard discriminator, based on text
    self.bert: standard transformer model with output logits. Bert for sequence classification head applied with 3
        outputs.

    forward:
        input:
            attention mask: standard transformer attention mask
            input_one_hot: the input sentence as a one hot vector. converted this way because the generator outputs
                a one hot vector. The two generator outputs as well as the target are randomly permuted at the beginning
                and are separated by sep tokens.
            masked_token_pos: Unused. Kept for similarity with sbert_sub model
        output:
            a probability array of 3 values, indicating whether the corresponding word position is assumed to be the
            original.
    """
    def __init__(self):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained(os.path.join(os.getcwd(), 'bert-base-uncased'),
            num_labels=3)

    def forward(self, attention_mask, input_one_hot, masked_token_pos=None):
        embeddings = torch.matmul(input_one_hot, self.bert.get_input_embeddings().weight)
        bert_output = self.bert(inputs_embeds=embeddings, attention_mask=attention_mask).logits
        return F.softmax(bert_output, dim=-1)


class Generator(LightningModule):
    """
        Nearly identical to generator from sbert_sub
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
    """

    def __init__(self):
        super().__init__()
        self.bert = BertForMaskedLM.from_pretrained(os.path.join(os.getcwd(), 'bert-base-uncased'))

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


class LexModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.Disc = Discriminator()
        self.Gen_A = Generator()
        if hparams.TYPE == "normal":
            self.Gen_B = Generator()
        elif hparams.TYPE == 'ema':
            self.Gen_B = EMA(self.Gen_A, hparams.DECAY)
        elif hparams.TYPE == 'frozen':
            self.Gen_B.freeze()
        self.epoch = 0

    '''
    Passes inputs into the discriminator which are shuffled.
    '''

    def forward(self, out_a, out_b, attention_mask_disc, masked_token_one_hot, masked_sent_ids,
            masked_sent_one_hot, batch_sep_token_one_hot, rand_perm, masked_token_pos):
        one_hot_cat = torch.cat([masked_token_one_hot, out_a, out_b], dim=-2)
        one_hot_shuffled = one_hot_cat.gather(1,
            torch.zeros(one_hot_cat.shape, dtype=torch.int64).type_as(masked_sent_ids) + rand_perm.reshape(1, -1, 1))
        disc_input = torch.cat([masked_sent_one_hot[:, 0:1, :], one_hot_shuffled[:, 0:1, :], batch_sep_token_one_hot,
            one_hot_shuffled[:, 1:2, :],
            batch_sep_token_one_hot, one_hot_shuffled[:, 2:3, :], batch_sep_token_one_hot,
            masked_sent_one_hot[:, 1:, :]], dim=-2)
        return self.Disc(attention_mask_disc, disc_input, masked_token_pos + 6)

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
        pass the padded sentence ids through the generators to receive substitute tokens
        All discriminator steps are run through the forward command. The forward command shuffles the target word, out_a
            and out_b (to prevent discrimination based on position) and then passes the input into the discriminator.
        if a generator step: train the generator to minimize discriminator accuracy and sbert distance. Loss 
            relationship given by alpha
        if a discriminator step: train the discriminator to correctly identify generated tokens
            - previous versions of this worked with two discriminators. Toggled by setting TYPE
    '''

    def training_step(self, batch, batch_idx, optimizer_idx):
        attention_mask, masked_sent_ids, padded_sent_ids, masked_token_id, masked_token_pos = batch
        masked_sent_one_hot = F.one_hot(masked_sent_ids, tokenizer.vocab_size).type_as(masked_sent_ids)
        masked_token_one_hot = F.one_hot(masked_token_id, tokenizer.vocab_size).type_as(masked_token_id)
        batch_sep_token_one_hot = (
                torch.zeros(masked_token_one_hot.shape) + F.one_hot(torch.tensor(tokenizer.sep_token_id),
            tokenizer.vocab_size)).type_as(masked_token_one_hot)
        attention_mask_gen = torch.cat([torch.ones(attention_mask.shape[0], 2).type_as(attention_mask), attention_mask],
            dim=-1)
        attention_mask_disc = torch.cat(
            [torch.ones(attention_mask.shape[0], 6).type_as(attention_mask), attention_mask],
            dim=-1)
        rand_perm = torch.randperm(3, dtype=torch.int64).type_as(
            masked_sent_ids)  # Same permutation for the entire batch. Doesn't matter in the long run.

        out_a = self.Gen_A(attention_mask_gen, padded_sent_ids*1, masked_token_one_hot, masked_token_pos + 1)
        out_b = self.Gen_B(attention_mask_gen, padded_sent_ids*1, masked_token_one_hot, masked_token_pos + 1, out_a.detach())

        classes = torch.eye(3, dtype=torch.int64).type_as(masked_sent_ids)
        output_class_shuffled = classes.gather(0,
            torch.zeros(classes.shape, dtype=torch.int64).type_as(masked_sent_ids) + rand_perm.view(-1, 1))

        if optimizer_idx == 1:
            if hparams.TYPE == 'ema':
                self.Gen_B.update()
            a_logits = self(out_a, out_b.detach(), attention_mask_disc, masked_token_one_hot, masked_sent_ids,
                masked_sent_one_hot, batch_sep_token_one_hot, rand_perm, masked_token_pos)
            non_b_idx = (rand_perm != 2).nonzero().view(-1).tolist()
            output = torch.zeros(a_logits.shape).type_as(a_logits) + output_class_shuffled[:, 1:2].reshape(1, -1)
            gen_a_loss = F.binary_cross_entropy(a_logits[:, non_b_idx], output[:, non_b_idx]) / math.log(2)
            self.log('gen_a_loss', gen_a_loss, prog_bar=True, logger=True)
            return gen_a_loss

        if optimizer_idx == 2:
            b_logits = self(out_a.detach(), out_b, attention_mask_disc, masked_token_one_hot, masked_sent_ids,
                masked_sent_one_hot, batch_sep_token_one_hot, rand_perm, masked_token_pos)
            non_a_idx = (rand_perm != 1).nonzero().view(-1).tolist()
            output = torch.zeros(b_logits.shape).type_as(b_logits) + output_class_shuffled[:, 2:3].reshape(1, -1)
            gen_b_loss = F.binary_cross_entropy(b_logits[:, non_a_idx], output[:, non_a_idx]) / math.log(2)
            self.log('gen_b_loss', gen_b_loss, prog_bar=True, logger=True)
            return gen_b_loss

        disc_logits = self(out_a.detach(), out_b.detach(), attention_mask_disc, masked_token_one_hot, masked_sent_ids,
            masked_sent_one_hot, batch_sep_token_one_hot, rand_perm, masked_token_pos)
        disc_loss = F.binary_cross_entropy(disc_logits,
            torch.zeros(disc_logits.shape).type_as(disc_logits) + output_class_shuffled[:, 0:1].reshape(1,
                -1)) / math.log(2)
        self.log('disc_loss', disc_loss, prog_bar=True, logger=True)
        return disc_loss

    def configure_optimizers(self):
        Disc_opt = torch.optim.Adam([
            {"params": self.Disc.parameters()},
        ], lr=hparams.DISC_LEARNING_RATE)
        Gen_A_opt = torch.optim.Adam([
            {"params": self.Gen_A.parameters()},
        ], lr=hparams.GEN_LEARNING_RATE)
        Gen_B_opt = torch.optim.Adam([
            {"params": self.Gen_B.parameters()},
        ], lr=hparams.GEN_LEARNING_RATE)
        if hparams.TYPE == 'normal':
            return [
                {'optimizer': Disc_opt},# 'frequency': 1},
                {'optimizer': Gen_A_opt},# 'frequency': 0},
                {'optimizer': Gen_B_opt},# 'frequency': 0},
            ]
        else:
            return [
                {'optimizer': Disc_opt},# 'frequency': 1},
                {'optimizer': Gen_A_opt},# 'frequency': 0},
            ]

    def training_epoch_end(self, batch):
        swords_gen(self, self.epoch)
        self.epoch += 1

    def generate(self, attention_mask, padded_sent_ids, masked_token_id, masked_token_pos, use_b=False):
        masked_token_one_hot = F.one_hot(masked_token_id, tokenizer.vocab_size).type_as(masked_token_id)
        if use_b:
            return self.Gen_B(attention_mask, padded_sent_ids, masked_token_one_hot, masked_token_pos + 1, softmax=True)
        return self.Gen_A(attention_mask, padded_sent_ids, masked_token_one_hot, masked_token_pos + 1, softmax=True)

    def playground_forward(self, attention_mask, masked_sent_ids, padded_sent_ids, masked_token_id, masked_token_pos):
        # Method used for testing and iteration alongside playground script. Not needed for training
        masked_sent_one_hot = F.one_hot(masked_sent_ids, tokenizer.vocab_size).type_as(masked_sent_ids)
        masked_token_one_hot = F.one_hot(masked_token_id, tokenizer.vocab_size).type_as(masked_token_id)
        batch_sep_token_one_hot = (
                torch.zeros(masked_token_one_hot.shape) + F.one_hot(torch.tensor(tokenizer.sep_token_id),
            tokenizer.vocab_size)).type_as(masked_token_one_hot)
        attention_mask_gen = torch.cat([torch.ones(attention_mask.shape[0], 2).type_as(attention_mask), attention_mask],
            dim=-1)
        attention_mask_disc = torch.cat(
            [torch.ones(attention_mask.shape[0], 6).type_as(attention_mask), attention_mask],
            dim=-1)
        rand_perm = torch.tensor([0, 1, 2], dtype=torch.int64).type_as(
            masked_sent_ids)  # Same permutation for the entire batch. Doesn't matter

        out_a_soft = self.Gen_A(attention_mask_gen, padded_sent_ids, masked_token_one_hot, masked_token_pos + 1,
            softmax=True)
        out_a = F.one_hot(torch.argmax(out_a_soft, dim=-1), num_classes=tokenizer.vocab_size)
        out_b_soft = self.Gen_B(attention_mask_gen, padded_sent_ids, masked_token_one_hot, masked_token_pos + 1, out_a=out_a.detach()*1, softmax=True)
        out_b = F.one_hot(torch.argmax(out_b_soft, dim=-1), num_classes=tokenizer.vocab_size)

        classes = torch.eye(3, dtype=torch.int64).type_as(masked_sent_ids)
        output_class_shuffled = classes.gather(0,
            torch.zeros(classes.shape, dtype=torch.int64).type_as(masked_sent_ids) + rand_perm.view(-1, 1))

        one_hot_cat = torch.cat([masked_token_one_hot, out_a, out_b], dim=-2)
        one_hot_shuffled = one_hot_cat.gather(1,
            torch.zeros(one_hot_cat.shape, dtype=torch.int64).type_as(masked_sent_ids) + rand_perm.reshape(1, -1, 1))
        disc_input = torch.cat([masked_sent_one_hot[:, 0:1, :], one_hot_shuffled[:, 0:1, :], batch_sep_token_one_hot,
            one_hot_shuffled[:, 1:2, :],
            batch_sep_token_one_hot, one_hot_shuffled[:, 2:3, :], batch_sep_token_one_hot,
            masked_sent_one_hot[:, 1:, :]], dim=-2)

        disc_logits = self.Disc(attention_mask_disc, disc_input.type_as(torch.zeros(1, dtype=torch.float).to('cpu')),
            masked_token_pos + 6)

        print(tokenizer.decode(torch.argmax(disc_input, dim=-1)[0]))
        print(tokenizer.decode(torch.argmax(disc_logits)))
        return disc_logits, out_a, out_b, output_class_shuffled, out_a_soft, out_b_soft


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

        # check if both model contains the same set of keys
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
            # buffers are copied
            shadow_buffers[name].copy_(buffer)

    def forward(self, attention_mask, input_ids, masked_token_one_hot, masked_token_pos, out_a=0, softmax=False):
        return self.shadow(attention_mask, input_ids, masked_token_one_hot, masked_token_pos, out_a, softmax)  # We want the shadow while training as well
