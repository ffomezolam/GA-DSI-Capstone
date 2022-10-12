""" transformers-model.py
-------------------------
Script to fit a gpt2 transformers model. Intended for use via cloud computing.
"""

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import TFAutoModelForCausalLM, AdamWeightDecay

from itertools import chain
import os
import random
import re
import json
import sys
import argparse

MODEL_TYPE = 'gpt2-medium'
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else 'test'
MODEL_DIR = './models/'
DATA_DIR = './data/'

RE_SENTENCE = re.compile(r'\w.*?[.?!:;]', re.S)
RE_WHITESPACE = re.compile(r'\s+')
RE_BLANKLINE = re.compile(r'^\s*$')

END_OF_LINE_TOKEN = '<|eol|>'
END_OF_SENTENCE_TOKEN = '<|eos|>'

#--- ARGUMENTS --------------------------------------------------------------

argp = argparse.ArgumentParser(description="create and fit gpt2 model")

args = argp.parse_args()
