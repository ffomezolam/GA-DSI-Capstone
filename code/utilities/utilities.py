""" utilities.py
-------------------------
Support definitions
"""

from transformers import AutoTokenizer
from transformers import TFAutoModelForCausalLM, TFAutoModelForSequenceClassification

from sklearn.model_selection import train_test_split

import numpy as np

import re
import os
import random
import json

#--- DATA SUPPORT

def load_config(fn='config.json'):
    "Get specified config file from disk"
    if type(fn) == str:
        with open(fn, 'r') as jsf:
            return json.load(jsf)
    else:
        # assume it's an already-loaded config if not string
        return fn

def load_text_from_config(config='config.json'):
    "Load data from config object or file"
    config = load_config(config)
    shakespeare = load_text_files(config['DATA_SHAKESPEARE'], config['DATA_DIR'])
    other = load_text_files(config['DATA_OTHER'], config['DATA_DIR'])
    return shakespeare, other

def get_labels(yes_data, no_data):
    "Apply category labels to data by turning lists into 2d arrays"
    yes_labels = [1 for _ in yes_data]
    no_labels = [0 for _ in no_data]
    return yes_labels, no_labels

def load_text_files(fns, data_dir):
    "Load and join all files in `fns` list"
    data = list()
    if type(fns) == str: fns = [fns]
    for fn in fns:
        path = os.path.join(data_dir, fn)
        with open(path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            data.append(' '.join(lines))

    return '\n'.join(data)

def get_dataset_from_config(config='config.json', limit=1):
    """
    Get dataset in format [[(text, label_0), ...],[(text, label_1), ...]]
    from config object or file
    """
    shakespeare, other = load_text_from_config(config)
    shakespeare = extract_sentences(shakespeare)
    other = extract_sentences(other)

    # limiit has been specified
    if limit > 0 and limit != 1:
        s_len = len(shakespeare)
        o_len = len(other)

        # limit is ratio
        if limit < 1:
            shakespeare = shakespeare[:int(s_len * limit)]
            other = other[:int(o_len * limit)]

        # limit is absolute
        elif limit > 1:
            tot = s_len + o_len
            s_rat = s_len / tot
            o_rat = o_len / tot
            s_lim = int(limit * s_rat)
            o_lim = int(limit * o_rat)
            shakespeare = shakespeare[:s_lim]
            other = other[:o_lim]

    s_labels, o_labels = get_labels(shakespeare, other)
    return [list(zip(other, o_labels)), list(zip(shakespeare, s_labels))]

def train_test_val_split(data, test_ratio=0.1):
    "custom train test split to add validation split"
    splits = {'train': list(),
              'test': list(),
              'val': list()}

    val_ratio = test_ratio * test_ratio

    len0 = len(data[0])
    len1 = len(data[1])
    test_len0 = int(len0 * test_ratio) or 1
    test_len1 = int(len1 * test_ratio) or 1
    val_len0 = int(len0 * val_ratio) or 1
    val_len1 = int(len1 * val_ratio) or 1

    sampled_0 = random.sample(data[0], len0)
    sampled_1 = random.sample(data[1], len1)
    sampled_0_test = sampled_0[:test_len0]
    sampled_0 = sampled_0[test_len0:]
    sampled_1_test = sampled_1[:test_len1]
    sampled_1 = sampled_1[test_len1:]
    sampled_0_val = sampled_0[:val_len0]
    sampled_0 = sampled_0[val_len0:]
    sampled_1_val = sampled_1[:val_len1]
    sampled_1 = sampled_1[val_len1:]

    train = sampled_0 + sampled_1
    test = sampled_0_test + sampled_1_test
    val = sampled_0_val + sampled_1_val

    return {
        'train': random.sample(train, len(train)),
        'test': random.sample(test, len(test)),
        'val': random.sample(val, len(val))
    }

def split_text_and_labels(ds):
    text = [i[0] for i in ds]
    labels = [i[1] for i in ds]
    return {'text': text, 'labels': labels}

#--- TEXT PROCESSING

## Useful regexes
RE_WORD = re.compile(r"\b[\w'’]+\b")
RE_WHITESPACE = re.compile(r'\s+')
RE_SENTENCE = re.compile(r'\w.*?[.?!:;]', re.S)
RE_PUNCTUATION = re.compile(r'[.,?!;:]')
RE_BLANKLINE = re.compile(r'\n\n')

## functions
def extract_words(text, n=1):
    "split text into words based on regex"
    return RE_WORD.findall(text)

def extract_sentences(text):
    "split text into sentences based on regex"
    sentences =  RE_SENTENCE.findall(text)
    sentences = [RE_WHITESPACE.sub(' ', sentence) for sentence in sentences]
    return sentences

#--- MODEL SUPPORT

def get_model_path(config='config.json', model_type='causal'):
    "get model path from config"
    cfgvars = load_config(config)

    mtype = 'CAUSAL' if model_type == 'causal' else 'CLASS'

    dir = cfgvars['MODEL_DIR']
    name = cfgvars['MODEL_NAME']
    type = cfgvars[f'{mtype}_MODEL']
    epochs = cfgvars[f'{mtype}_N_EPOCHS']

    model_dirname = f'{name}.{type}.{epochs}'
    return os.path.join(dir, model_dirname)

def load_model(path, type='causal'):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Model path not found: {path}')

    if type == 'causal':
        return TFAutoModelForCausalLM.from_pretrained(path)
    else:
        return TFAutoModelForSequenceClassification.from_pretrained(path)

def load_tokenizer(type):
    return AutoTokenizer.from_pretrained(type)

def generate_from(text, model, tokenizer,
                  max=100,
                  temp=1,
                  k=50,
                  rep_penalty=1.5,
                  len_penalty=0.75,
                  n_seq=1):
    tokens = tokenizer(text, return_tensors='tf')
    output = model.generate(**tokens,
                            do_sample=True,
                            max_new_tokens=max,
                            temperature=temp,
                            top_k=k,
                            repetition_penalty=rep_penalty,
                            length_penalty=len_penalty,
                            num_return_sequences=n_seq)
    return tokenizer.decode(output[0], skip_special_tokens=True)

class ClassificationResult:
    def __init__(self, text, outputs):
        self.text = text
        self.outputs = outputs
        self.c = self.classifications = self.classify()
        self.p = self.s = self.scores = self.probs = self.get_prob()

    def classify(self):
        return np.argmax(self.outputs.logits, axis=1)

    def get_prob(self):
        logits = self.outputs.logits
        return (np.exp(logits) / (1 + np.exp(logits)))[:,1]

    def get_results(self):
        return list(zip(self.text, self.c, self.s))

    def __iter__(self):
        return zip(self.text, self.c, self.s)

def classify_from(text, model, tokenizer):
    tokens = tokenizer(text, return_tensors='tf', padding=True)
    output = model(tokens)
    return ClassificationResult(text, output)

###------------------------------------------------------------- SELF-TEST

if __name__ == '__main__':
    import lorem
    import sys

    def separate():
        print(f'\n --- \n')

    if len(sys.argv) > 1:
        # argument
        with open(sys.argv[1], 'r') as f:
            text = f.read()

        if len(sys.argv) > 2:
            n = int(sys.argv[2])
            if n < 0:
                text = text[n:]
            else:
                text = text[:n]
    else:
        text = ' '.join([lorem.paragraph() for _ in range(4)])

    separate()
    print(text)
    separate()
    print(extract_words(text))
    separate()
    print(extract_sentences(text))
    separate()
