import os
import hparams
import random
rng = random.Random(hparams.seed)


def rel_path(path):
    return os.path.join('data', path)


"""
Splits a large wikipedia paragraph into multiple lines between min_length and max_length words.
Prioritizes splitting on newlines, followed by periods, followed by spaces 
"""


in_file = open(rel_path("wikipedia_out.txt"), 'r')
train_txt = open(rel_path("train_long.txt"), 'w')
valid_txt = open(rel_path("valid_long.txt"), 'w')
test_txt = open(rel_path("test_long.txt"), 'w')
train_msg = open(rel_path("train_long.msg.txt"), 'w')
valid_msg = open(rel_path("valid_long.msg.txt"), 'w')
test_msg = open(rel_path("test_long.msg.txt"), 'w')


min_length = 160
max_length = 200
TRAIN_PERCENT = .9
VALID_PERCENT = .05
TEST_PERCENT = .05
out_txt = []
for line in in_file:
    if len(out_txt) % 1000 == 0:
        print(len(out_txt)//1000)
        if len(out_txt)//1000 > 100000: #ends the code early
            break
    while(len(line) > max_length*1.2):
        count_space = 0
        best_new_line = -1
        best_period = -1
        last_space = -1
        for i in range(len(line)):
            if line[i] == " ":
                count_space += 1
                last_space = i
                if count_space > max_length:
                    break
            elif line[i] == "\n" and count_space >= min_length:
                best_new_line = i
                break
            elif i < len(line)-1 and line[i:i+2] == ". " and count_space >= min_length:
                best_period = i
                break
        if best_new_line > -1:
            out_txt.append(line[:best_new_line+1])
            line = line[best_new_line+1:]
        elif best_period > -1:
            out_txt.append(line[:best_period+1])
            line = line[best_period+1:]
        elif count_space > min_length:
            out_txt.append(line[:last_space+1])
            line = line[last_space+1:]
        else:
            break

rng.shuffle(out_txt)

out_msg = []
for i in range(len(out_txt)):
    st = ""
    for j in range(32):
        st += str(rng.randint(0, 1))
    out_msg.append(st)

for i in range(int(len(out_txt)*TRAIN_PERCENT)):
    train_txt.write(out_txt[i].strip() + "\n")
    train_txt.flush()
    train_msg.write(out_msg[i].strip() + "\n")
    train_msg.flush()

for i in range(int(len(out_txt)*TRAIN_PERCENT), int(len(out_txt)*(TRAIN_PERCENT+VALID_PERCENT))):
    valid_txt.write(out_txt[i].strip() + "\n")
    valid_txt.flush()
    valid_msg.write(out_msg[i].strip() + "\n")
    valid_msg.flush()

for i in range(int(len(out_txt)*(TRAIN_PERCENT+VALID_PERCENT)), len(out_txt)):
    test_txt.write(out_txt[i].strip() + "\n")
    test_txt.flush()
    test_msg.write(out_msg[i].strip() + "\n")
    test_msg.flush()