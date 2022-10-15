#!python
""" generate.py
---------------
Generate shakespearean text
"""

# model
#from transformers import AutoTokenizer, TFAutoModelForCausalLM

# custom utils
from utilities.utilities import load_config
from utilities.utilities import get_model_path, load_tokenizer, load_model
from utilities.utilities import generate_from
from utilities.utilities import extract_sentences

if __name__ == '__main__':
    import argparse
    import sys
    import os

    #--- ARGUMENTS
    argp = argparse.ArgumentParser(description="Generate Shakespearean text")

    argp.add_argument('infile', nargs='?', type=argparse.FileType('r'),
                      default=sys.stdin)

    argp.add_argument('outfile', nargs='?', type=argparse.FileType('w'),
                      default=sys.stdout)

    argp.add_argument('-c', '--config', type=str, default='config-test.json',
                      help="config file path")

    argp.add_argument('-s', '--sentence', action='count', default=0,
                      help="Only use first sentence of output")

    args = argp.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f'Config file not found: {args.config}')

    cfg = load_config(args.config)
    model_path = get_model_path(cfg)

    tokenizer = load_tokenizer(cfg['CAUSAL_MODEL'])

    model = load_model(model_path)

    text = args.infile.read().split('\n')
    text = [line.strip() for line in text if line.strip()]

    gens = list()
    for line in text:
        gen = generate_from(line, model, tokenizer)
        if args.sentence: gen = extract_sentences(gen)[0]
        gens.append(gen)

    args.outfile.write('\n'.join(gens))
