""" utilities.py
-------------------------
Support definitions
"""

from transformers import AutoTokenizer, TFAutoModelForCausalLM

from sklearn.model_selection import train_test_split

import re
import os
import random
import json

#--- DATA SUPPORT

def load_config(fn='config.json'):
    "Get specified config file from disk"
    with open(fn, 'r') as jsf:
        return json.load(jsf)

def load_text_from_config(config='config.json'):
    "Load data from config object or file"
    if config[-5:] == '.json': config = load_config(config)
    shakespeare = load_text_files(config['DATA_SHAKESPEARE'], config['DATA_DIR'])
    other = load_text_files(config['DATA_OTHER'], config['DATA_DIR'])
    return shakespeare, other

def label_data(yes_data, no_data):
    "Apply category labels to data by turning lists into 2d arrays"
    yes_data = [[item, 1] for item in yes_data]
    no_data = [[item, 0] for item in no_data]
    return yes_data, no_data

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

def get_dataset_from_config(config='config.json'):
    "get prepped dataset from config object or file"
    shakespeare, other = load_text_from_config(config)
    shakespeare = extract_sentences(shakespeare)
    other = extract_sentences(other)
    shakespeare, other = label_data(shakespeare, other)
    return shakespeare + other

def train_val_test_split(data, test_ratio=0.1, shuffle=True):
    "wrapper for sklearn tts to add validation split"
    pass

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

def train_test_split(text, test_size=0.1, shuffle=False):
    pass

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
