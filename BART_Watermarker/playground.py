from model import MainModel
from data import encode_msgs
import hparams
import torch

"""
Playground file to evaluate watermarker.
Inputs a sentence and proceeds to encode it with every bit sequence
"""

model = MainModel.load_from_checkpoint('BART_Watermarker.ckpt')
watermarker = model.watermarker
msg_decoder = model.msg_decoder

while True:
    sent = input("Sentence\n")
    for i in range(2**hparams.MSG_LEN):
        bits = bin(i)[2:].zfill(hparams.MSG_LEN)
        print(bits)

        tokenized = hparams.tokenizer([sent], return_tensors='pt', padding=True, truncation=True, max_length=80, add_special_tokens=False)
        msg_compact, msg_one_hot = encode_msgs([bits])

        with torch.no_grad():
            wm_output, wm_one_hot, msg_output = model(tokenized.attention_mask, tokenized.input_ids, msg_compact)
            print(hparams.tokenizer.decode(torch.argmax(wm_output.logits, dim=-1).reshape(-1)))
            print(msg_output)
            print()
