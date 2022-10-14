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

def get_dataset_from_config(config='config.json'):
    """
    Get dataset in format [[(text, label_0), ...],[(text, label_1), ...]]
    from config object or file
    """
    shakespeare, other = load_text_from_config(config)
    shakespeare = extract_sentences(shakespeare)
    other = extract_sentences(other)
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
