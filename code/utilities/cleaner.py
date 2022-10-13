#!python
""" cleaner.py
--------------
Utilities for cleaning up text
"""

#--- IMPORTS
import re

#--- HELPERS
def paren(string, parens='()'):
    "Returns string wrapped with parens"
    return parens[0] + string + parens[1]

def make_re_cc(string, negate=False):
    "Makes string into a character class"
    start = '[^' if negate else '['
    end = ']'
    return start + string + end

#--- REGEX STRINGS
QUOTES = {
    'all': '\'"’‘“”',
    'any': '\'"',
    'left': '\'"“‘',
    'right': '\'"’”',
}

CONTRACTIONS = {
    'e': {
        'start': 'r|v|s|p|w|m|y|l|b',
        'end': 'd|st'
    },
    'v': {
        'start': 'o',
        'end': 'e'
    }
}

#--- COMPILED REGEXES
RE_METADATA = {
    "gutenberg": {
        "header": {
            "re": re.compile(r'\*{3}\s*START.+\s*\*{3}'),
            "discard": 'pre'
        },
        "footer": {
            "re": re.compile(r'\*{3}\s*END.+\s*\*{3}'),
            "discard": 'post'
        }
    }
}

RE_QUOTES = {key: re.compile(QUOTES[key]) for key in QUOTES}

RE_CONTRACTIONS = {letter: re.compile(paren(CONTRACTIONS[letter]['start'])\
                                    + paren(make_re_cc(QUOTES['right']))\
                                    + paren(CONTRACTIONS[letter]['end']))\
                   for letter in CONTRACTIONS}

RE_QUOTATION = {
    'open': re.compile(r'(\s*)' + make_re_cc(QUOTES['left'])),
    'close': re.compile(r'(?!s)' + make_re_cc(QUOTES['right']) + r'(\s*)')
}

RE_EMPH = re.compile(r'_(\w+)_')

RE_REF = re.compile(r'\[\w+]')
RE_FOOTNOTES = re.compile(r'\s*FOOTNOTES\W*', re.I)
RE_BLANKLINE = re.compile(r'\n+\W*\n+')

#--- CONTENT ISOLATION

def strip_metadata(text, type="gutenberg", verbosity=0):
    v = verbosity
    RES = {k: v for k, v in RE_METADATA[type].items()}
    for type, data in RES.items():
        RE = data['re']
        discard = data['discard']
        if v > 0: print(f'discarding {type}')
        split = RE.split(text)

        if discard == 'pre':
            text = split[1]
        elif discard == 'post':
            text = split[0]

    return text.strip()

#--- TEXT CLEANING
def sub_contractions(text, c_type=None):
    """
    Substitute contractions. For example:
    1. o'er -> over
    2. diseas'd -> diseased
    """
    if not c_type: c_type = list(RE_CONTRACTIONS.keys())
    elif type(c_type) == str:
        c_type = [c_type]

    for t in c_type:
        if t not in RE_CONTRACTIONS: continue

        text = RE_CONTRACTIONS[t].sub(r'\1' + t + r'\3', text)

    return text

def strip_quotes(text):
    """
    Strip quotes. For example:
    1. 'monolith' -> monolith
    2. "haberdashery" -> haberdashery
    """
    text = RE_QUOTATION['open'].sub(r'\1', text)
    text = RE_QUOTATION['close'].sub(r'\1', text)
    return text

def remove_emph(text):
    return RE_EMPH.sub(r'\1', text)

def remove_references(text):
    text = RE_REF.sub('', text)
    split = RE_FOOTNOTES.split(text)
    return split[0].strip()

def remove_blanklines(text, replace=r'\n\n'):
    return RE_BLANKLINE.sub(replace, text)

def remove_leading_spaces(text):
    lines = text.split('\n')
    return '\n'.join([line.strip() for line in lines])

#--- CL CODE ----------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    import sys

    #--- Arguments ---
    argp = argparse.ArgumentParser(description="Clean up source text")

    argp.add_argument('infile', nargs='?', type=argparse.FileType('r'),
                      default=sys.stdin)
    argp.add_argument('outfile', nargs='?', type=argparse.FileType('w'),
                      default=sys.stdout)
    argp.add_argument('-v', '--verbose', action='count', default=0)
    argp.add_argument('-a', action='count', default=0,
                      help="full clean")
    argp.add_argument('-m', action='count', default=0,
                      help="remove metadata")
    argp.add_argument('-q', action='count', default=0,
                      help="remove quotation marks")
    argp.add_argument('-e', action='count', default=0,
                      help="remove emphasis")
    argp.add_argument('-b', action='count', default=0,
                      help="remove blank lines")
    argp.add_argument('-s', action='count', default=0,
                      help="remove leading spaces")

    args = argp.parse_args()

    #--- Process ---
    if args.verbose: print(f'Loading {args.infile.name} ...', end='')
    text = args.infile.read()
    if args.verbose: print(' Done')
    if args.verbose > 1: print(f'\n*--- SAMPLE ---*\n{text[:500]}\n*--- END ---*\n')

    if args.m or args.a:
        if args.verbose: print('Stripping metadata ... ', end='')
        text = strip_metadata(text)
        text = remove_references(text)
        if args.verbose: print(' Done')
        if args.verbose > 1: print(f'\n*--- SAMPLE ---*\n{text[:500]}\n*--- END ---*\n')

    if args.q or args.a:
        if args.verbose: print('Stripping quotation marks ... ', end='')
        text = strip_quotes(text)
        if args.verbose: print('Done')

    if args.e or args.a:
        if args.verbose: print('Removing emphasis ... ', end='')
        text = remove_emph(text)
        if args.verbose: print('Done')

    # there's a problem with this one so not including in 'all'
    if args.b:
        if args.verbose: print('Removing blank lines ... ', end='')
        text = remove_blanklines(text)
        if args.verbose: print('Done')

    if args.s or args.a:
        if args.verbose: print('Removing leading spaces ... ', end='')
        text = remove_leading_spaces(text)
        if args.verbose: print('Done')

    args.outfile.write(text)
