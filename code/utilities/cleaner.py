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
RE_QUOTES = {key: re.compile(QUOTES[key]) for key in QUOTES}

RE_CONTRACTIONS = {letter: re.compile(paren(CONTRACTIONS[letter]['start'])\
                                    + paren(make_re_cc(QUOTES['right']))\
                                    + paren(CONTRACTIONS[letter]['end']))\
                   for letter in CONTRACTIONS}

RE_QUOTATION = {
    'open': re.compile(r'(\s*)' + make_re_cc(QUOTES['left'])),
    'close': re.compile(r'(?!s)' + make_re_cc(QUOTES['right']) + r'(\s*)')
}

#--- MAIN UTILITY FUNCTIONS
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

#--- SELF-TEST --------------------------------------------------------------
if __name__ == '__main__':
    import sys

    path = sys.argv[1]
    with open(path, 'r') as f:
        lines = f.readlines()

    # substitute contractions
    for ix in range(len(lines)):
        lines[ix] = sub_contractions(lines[ix])

    # strip quotes
    for ix in range(len(lines)):
        lines[ix] = strip_quotes(lines[ix])

    print(''.join(lines))
