from model import LexModel
from util import tokenizer
import torch
from itertools import islice
import hparams
import random
rng = random.Random(hparams.seed)

'''
Runs through 100 random examples from the validation set and outputs:
    The word at the location of the generator head. Should be the original word.
    The sentence with sep tokens
    The random permutation (kept constant)
    The discriminator probabilities for the: Original, Gen A, Gen B
    The original word
    The output of Gen A
    The output of Gen B
    The list of top-k Gen A outputs (What SWORDS receives)
    The scores of top-k Gen A outputs (What SWORDS receives)
    The lost of top-k Gen B outputs
    The scores of top-k Gen B outputs
At the end, a sum of probabilities is also output
'''

num = 100

lines = [line.strip() for line in islice(open("data/valid_syn.txt", 'r'), 10000)]
rng.shuffle(lines)
masked_sent_ids = []
padded_sent_ids = []
masked_token_ids = []
masked_token_pos = []
attention_mask = []
for line in lines[:num]:
    start = line.find(" ")
    masked_token = line[:start]
    masked_token_id = tokenizer.encode(masked_token, add_special_tokens=False, return_tensors='pt')[0]
    if len(masked_token_id) > 1:
        num -= 1
        continue
    tokenized = tokenizer(line[start + 1:], return_tensors='pt', padding=True, pad_to_multiple_of=hparams.MAX_LENGTH,
        max_length=hparams.MAX_LENGTH, truncation=True)
    try:
        index = int((tokenized.input_ids == tokenizer.mask_token_id)[0].nonzero())
    except:
        num -= 1
        continue
    attention_mask.append(tokenized.attention_mask)
    masked_sent_ids.append(tokenized.input_ids)
    masked_token_pos.append([index])
    masked_token_ids.append([int(masked_token_id[0])])
    padded_sent_ids.append(torch.cat([tokenized.input_ids[0, :index], torch.tensor([1, masked_token_id, 1]),
        tokenized.input_ids[0, index + 1:]]).reshape(-1, hparams.MAX_LENGTH + 2))

attention_mask = torch.cat(attention_mask)
masked_sent_ids = torch.cat(masked_sent_ids)
padded_sent_ids = torch.cat(padded_sent_ids)
masked_token_ids = torch.tensor(masked_token_ids)
masked_token_pos = torch.tensor(masked_token_pos)


model = LexModel.load_from_checkpoint('lex_sub.ckpt')

sums = torch.zeros(3).to('cpu')

model = model.to('cpu')

while True:
    disc_logits, out_a, out_b, output_class_shuffled, out_a_soft, out_b_soft = model.playground_forward(attention_mask.to('cpu'), masked_sent_ids.to('cpu'), padded_sent_ids.to('cpu'), masked_token_ids.to('cpu'), masked_token_pos.to('cpu'))

    for i in range(num):
        sums += disc_logits[i]
        print(tokenizer.decode(masked_sent_ids[i]))
        print(disc_logits[i].tolist())
        print(output_class_shuffled[:, 0], output_class_shuffled[:, 1], output_class_shuffled[:, 2])
        print(tokenizer.decode(masked_token_ids[i]))
        print(tokenizer.decode(torch.argmax(out_a[i])))
        print(tokenizer.decode(torch.argmax(out_b[i])))
        print(torch.topk(out_a_soft[i][0], 5, dim=-1).values.tolist())
        print(tokenizer.decode(torch.topk(out_a_soft[i][0], 5, dim=-1).indices.tolist()))
        print(torch.topk(out_b_soft[i][0], 5, dim=-1).values.tolist())
        print(tokenizer.decode(torch.topk(out_b_soft[i][0], 5, dim=-1).indices.tolist()))
        print()
    break

print(sums)