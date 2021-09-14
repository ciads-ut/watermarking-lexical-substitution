from transformers import BertTokenizerFast, BertForMaskedLM, BertForSequenceClassification
import os

tokenizer = BertTokenizerFast.from_pretrained(os.path.join(os.getcwd(), 'bert-base-uncased'))

def create_bert_sequence(num_labels=3):
    return BertForSequenceClassification.from_pretrained(os.path.join(os.getcwd(), 'bert-base-uncased'), num_labels=num_labels)

def create_bert_MLM():
    return BertForMaskedLM.from_pretrained(os.path.join(os.getcwd(), 'bert-base-uncased'))
