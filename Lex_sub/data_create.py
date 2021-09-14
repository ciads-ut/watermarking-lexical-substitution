import os
import hparams
import random
rng = random.Random(hparams.seed)


def rel_path(path):
    return os.path.join('data', path)


'''
Takes in lines with the target word at the begging, having been replaced by a mask token. Splits data into train, val, test
'''

in_file = open(rel_path("wikipedia_out.lexsub.combined.txt"), 'r')
train_txt = open(rel_path("train_syn.txt"), 'w')
valid_txt = open(rel_path("valid_syn.txt"), 'w')
test_txt = open(rel_path("test_syn.txt"), 'w')


min_length = 80
max_length = 100
TRAIN_PERCENT = .9
VALID_PERCENT = .05
TEST_PERCENT = .05
out_txt = []
count = -1
prev_line = ""
for line in in_file:
    count += 1
    if count % 2 == 0:
        prev_line = line
        continue
    out_txt.append(prev_line.split()[0].strip() + " " + line.strip())

rng.shuffle(out_txt)

for i in range(int(len(out_txt)*TRAIN_PERCENT)):
    train_txt.write(out_txt[i].strip() + "\n")
    train_txt.flush()

for i in range(int(len(out_txt)*TRAIN_PERCENT), int(len(out_txt)*(TRAIN_PERCENT+VALID_PERCENT))):
    valid_txt.write(out_txt[i].strip() + "\n")
    valid_txt.flush()

for i in range(int(len(out_txt)*(TRAIN_PERCENT+VALID_PERCENT)), len(out_txt)):
    test_txt.write(out_txt[i].strip() + "\n")
    test_txt.flush()