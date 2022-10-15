#!python
""" classify.py
---------------
Classify as Shakespearean
"""
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

from utilities.utilities import load_config
from utilities.utilities import get_model_path, load_tokenizer, load_model
from utilities.utilities import classify_from

if __name__ == '__main__':
    import argparse
    import sys
    import os


    #--- ARGUMENTS
    argp = argparse.ArgumentParser(description="Generate Shakespearean text")

    argp.add_argument('infile', nargs='?', type=argparse.FileType('r'),
                      default=sys.stdin)

    argp.add_argument('-c', '--config', type=str, default='config-test.json',
                      help="config file path")

    args = argp.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f'Config file not found: {args.config}')

    cfg = load_config(args.config)
    model_path = get_model_path(cfg, 'class')

    tokenizer = load_tokenizer(cfg['CLASS_MODEL'])

    model = load_model(model_path, 'class')

    text = args.infile.read()
    text = [line.strip() for line in text.split('\n') if line.strip()]

    result = classify_from(text, model, tokenizer)

    for item in result:
        print(f'text: {item[0]}\nclass: {item[1]}\nscore: {item[2]}\n\n')
