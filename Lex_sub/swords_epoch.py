import gzip
import json
import warnings

k = 10

import re
import hparams
from util import *
import torch


def swords_gen(model, epoch):
    with gzip.open('swords-v1.1_dev.json.gz', 'r') as f:
        swords = json.load(f)

    def generate(
            context,
            target,
            target_offset,
            target_pos=None):
        """Produces _substitutes_ for _target_ span within _context_

        Args:
          context: A text context, e.g. "My favorite thing about her is her straightforward honesty.".
          target: The target word, e.g. "straightforward"
          target_offset: The character offset of the target word in context, e.g. 35
          target_pos: The UD part-of-speech (https://universaldependencies.org/u/pos/) for the target, e.g. "ADJ"

        Returns:
          A list of substitutes and scores e.g. [(sincere, 80.), (genuine, 80.), (frank, 70), ...]
        """
        context_1 = re.sub("\\s+", " ", context[:target_offset])
        context = re.sub("\\s+", " ", context)

        tokenized = tokenizer([context], return_tensors='pt')
        masked_token_id = tokenizer([target], add_special_tokens=False).input_ids[0]
        masked_token_id = masked_token_id[0]
        index = tokenized.char_to_token(len(context_1))
        full_padded = torch.cat([tokenized.input_ids[0, :index], torch.tensor([1, tokenized.input_ids[0, index], 1]),
            tokenized.input_ids[0, index + 1:]])
        start = max(1, index + 1 - hparams.MAX_LENGTH // 2)
        end = min(start + hparams.MAX_LENGTH, len(full_padded) - 1)
        padded_sent_ids = torch.tensor(
            [tokenizer.cls_token_id] + full_padded[start:end].tolist() + [tokenizer.sep_token_id] + (
                    torch.zeros((hparams.MAX_LENGTH - len(full_padded[start:end].tolist())),
                        dtype=torch.long) + tokenizer.pad_token_id).tolist()).reshape(-1, hparams.MAX_LENGTH + 2)
        index = hparams.MAX_LENGTH // 2 if start > 1 else index
        out = model.generate(torch.ones(padded_sent_ids.shape).to("cuda:1"), padded_sent_ids.to("cuda:1"),
            torch.tensor(masked_token_id).to("cuda:1"), torch.tensor(index).to("cuda:1"))
        scores = torch.topk(out[0][0], k, dim=-1).values.tolist()
        substitutes = tokenizer.batch_decode(torch.topk(out[0][0], k, dim=-1).indices.tolist())
        return list(zip(substitutes, scores))

    # NOTE: 'substitutes_lemmatized' should be True if your method produces lemmas (e.g. "run") or False if your method produces wordforms (e.g. "ran")
    result = {'substitutes_lemmatized': False, 'substitutes': {}}
    errors = 0
    for tid, target in swords['targets'].items():
        context = swords['contexts'][target['context_id']]
        try:
            result['substitutes'][tid] = generate(
                context['context'],
                target['target'],
                target['offset'],
                target_pos=target.get('pos'))
        except:
            errors += 1
            break


    if errors > 0:
        warnings.warn(f'{errors} targets were not evaluated due to errors')

    with open('swords-v1.lex_sub_ep_' + str(epoch) + '.val.json', 'w') as f:
        print()
        f.write(json.dumps(result))
