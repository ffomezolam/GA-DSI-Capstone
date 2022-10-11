""" word_deconstructor.py
-------------------------
Operations for deconstructing text into words
"""

import re
from transformers import AutoTokenizer, TFAutoModelForCausalLM

#--- TEXT PROCESSING

RE_WORD = re.compile(r"\b[\w'’]+\b")
RE_WHITESPACE = re.compile(r'\s+')
RE_SENTENCE = re.compile(r'\w.*?[.?!]', re.S)

def extract_words(text, n=1):
    return RE_WORD.findall(text)

def extract_sentences(text):
    return RE_SENTENCE.findall(text)

#--- MODEL SUPPORT

def make_tokenizer(type):
    return AutoTokenizer.from_pretrained(type)

def make_model(type):
    return TFAutoModelForCausalLM.from_pretrained(type)

def generate_from(text, model, tokenizer, max=100, temp=1, k=50, rep_penalty=1.5, len_penalty=0.75, n_seq=1):
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
