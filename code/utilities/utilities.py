""" word_deconstructor.py
-------------------------
Operations for deconstructing text into words
"""

import re
from transformers import AutoTokenizer, TFAutoModelForCausalLM
from rbapi import get_rhymes as rbrhymes

#--- TEXT PROCESSING

RE_WORD = re.compile(r"\b[\w'’]+\b")
RE_WHITESPACE = re.compile(r'\s+')
RE_SENTENCE = re.compile(r'\w.*?[.?!]', re.S)
RE_PUNCTUATION = re.compile(r'[.,?!;:]')

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

#--- RHYMING

def get_rhymes(blob):
    # convert to list
    if not type(blob) == list:
        blob = [line.strip() for line in blob.split('\n') if line.strip()]

    blob = [line.lower() for line in blob]

    final_words = [line.split()[-1] for line in blob]
    final_words = [word[:-1] if RE_PUNCTUATION.match(word[-1]) else word for word in final_words]

    rhymes = dict()

    for ix in range(len(final_words)):
        iword = final_words[ix]
        rhymes[iword] = list()
        irhymes = rbrhymes(iword)
        for jx in range(len(final_words)):
            if ix == jx: continue
            jword = final_words[jx]
            is_rhyme = (jword in irhymes) or (suffix_similarity(iword, jword) > 2)
            rhymes[iword].append((is_rhyme, jword))

    return rhymes

def suffix_similarity(word1, word2):
    # brute force rhyme test based on concluding letter similarity
    word1 = word1.lower()
    word2 = word2.lower()

    count = 0

    for i in range(len(word1)):
        if i >= len(word2): break
        ix = -i
        if word1[ix] == word2[ix]: count += 1

    return count
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
