""" transformers-model.py
-------------------------
Script to fit a gpt2 transformers model. Intended for use via cloud computing.
"""

from itertools import chain
import os
import random
import re
import json
import sys
import argparse
from datetime import datetime

MODEL_TYPE = 'distilgpt2'
MODEL_NAME = 'test'
MODEL_DIR = './models/'
NUM_EPOCHS = 8

MODEL_PATH = os.path.join(MODEL_DIR, f'{MODEL_TYPE}.{MODEL_NAME}.{str(NUM_EPOCHS)}')

RE_SENTENCE = re.compile(r'\w.*?[.?!:;]', re.S)
RE_WHITESPACE = re.compile(r'\s+')
RE_BLANKLINE = re.compile(r'\n\n+')

TEST_SIZE = 0.05

#--- ARGUMENTS --------------------------------------------------------------

argp = argparse.ArgumentParser(description="create and fit gpt2 models")

argp.add_argument('infile', nargs='?', type=argparse.FileType('r'),
                  default=sys.stdin)

argp.add_argument('--deploy', action='count', default=0)
argp.add_argument('-e', '--epochs', type=int, default=1)
argp.add_argument('--modeldir', type=str, default=MODEL_DIR)
argp.add_argument('--testsize', type=float, default=0.05)
argp.add_argument('-v','--verbose', action='count', default=0)

args = argp.parse_args()

NUM_EPOCHS = args.epochs
MODEL_DIR = args.modeldir
TEST_SIZE = args.testsize

#--- DATA -------------------------------------------------------------------

from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

text = args.infile.read()

input = {
    'poems': RE_BLANKLINE.split(text),
    'lines': [line.strip() for line in text.split('\n') if line.strip()],
    'sentences': [RE_WHITESPACE.sub(' ', line) for line in RE_SENTENCE.findall(text)]
}

if args.verbose:
    print(f'Poems: {len(input["poems"])}')
    print(f'Sentences: {len(input["sentences"])}')
    print(f'Lines: {len(input["lines"])}')

train_src = dict()
test_src = dict()

for k, v in input.items():
    train_src[k], test_src[k] = train_test_split(v, test_size=TEST_SIZE)

if args.verbose:
    print("TRAIN")
    for k, v in train_src.items():
        print(f'{k}: {len(v)}')

    print("TEST")
    for k, v in test_src.items():
        print(f'{k}: {len(v)}')

train = dict()
test = dict()

for k, v in train_src.items():
    train[k] = Dataset.from_dict({'text': v})

for k, v in test_src.items():
    test[k] = Dataset.from_dict({'text': v})

sets = dict()

for k in train:
    sets[k] = DatasetDict({'train': train[k], 'test': test[k]})

if args.verbose: print(sets)

#--- TOKENIZE ---------------------------------------------------------------

from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling

tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE)

def token_preproc(data):
    return tokenizer(data['text'])

tokenized = dict()

for k, v in sets.items():
    tokenized[k] = v.map(token_preproc, batched=True, num_proc=4, remove_columns=['text'])

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors='tf')

#--- MODEL ------------------------------------------------------------------

from transformers import TFAutoModelForCausalLM, AdamWeightDecay

if args.deploy:
    model = TFAutoModelForCausalLM.from_pretrained(MODEL_TYPE, pad_token_id=tokenizer.eos_token_id)
    optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
    model.compile(optimizer=optimizer)

    tokenizer.pad_token = tokenizer.eos_token
    model_sets = dict()
    for k, v in tokenized.items():
        model_sets[k] = {
            'train': model.prepare_tf_dataset(tokenized[k]['train'], shuffle=True, batch_size=32, collate_fn=collator),
            'test': model.prepare_tf_dataset(tokenized[k]['test'], shuffle=False, batch_size=32, collate_fn=collator)
        }

    for k,v in model_sets.items():
        model.fit(v['train'], validation_data=v['test'], epochs=NUM_EPOCHS)
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        model.save_pretrained(MODEL_PATH)
