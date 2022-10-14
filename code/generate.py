#!python
""" generate.py
---------------
Generate shakespearean text
"""

# model
from transformers import AutoTokenizer, TFAutoModelForCausalLM

# custom utils
from utilities.utilities import get_model_path, load_tokenizer, load_model
from utilities.utilities import generate_from

if __name__ == '__main__':
    import argparse
    import sys

    #--- ARGUMENTS
    argp = argparse.ArgumentParser(description="Generate Shakespearean text")

    argp.add_argument('infile', nargs='?', type=argparse.FileType('r'),
                      default=sys.stdin)

    argp.add_argument('outfile', nargs='?', type=argparse.FileType('w'),
                      default=sys.stdout)

    argp.add_argument('-c', '--config', type=str, default='config.json',
                      help="config file path")

    args = argp.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f'Config file not found: {args.config}')

    model_path = get_model_path(args.config)

    tokenizer = load_tokenizer(model_path)
    model = load_model(model_path)
