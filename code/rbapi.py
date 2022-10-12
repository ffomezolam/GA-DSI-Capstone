""" rbapi.py
------------
API wrapper for rhymebrain
"""

import requests
import json

BASE_URL = r'https://rhymebrain.com/talk?function=<FUNC>&word=<WORD>'
ADD_MAX_RESULTS = r'&maxResults=<NUM>'

functions = {
    'rhymes': 'getRhymes',
    'info': 'getWordInfo',
    'portmanteaus': 'getPortmanteaus'
}

def geturl(word, func='rhymes', num=None):
    "Returns the formatted API URL"
    url = BASE_URL.replace('<FUNC>', functions[func]).replace('<WORD>', word)
    if num: url += ADD_MAX_RESULTS.replace('<NUM>', num)
    return url

def get(word, func='rhymes', num=None, score=0, words_only=False, as_json=False):
    """
    Call the API
    """
    response = requests.get(geturl(word, func, num))
    result = json.loads(response.text)
    if score:
        result = [r for r in result if int(r['score'] >= score)]
    if words_only:
        result = [r['word'] for r in result]
    if not as_json: return result

    return json.dumps(result).strip()

def get_rhymes(word, num=None, score=0, words_only=True, as_json=False):
    "Wrapper for get(func='rhymes')"
    return get(word, 'rhymes', num, score, words_only, as_json)

def get_word_info(word, as_json=False):
    "Wrapper for get(func='info')"
    return get(word, 'info', as_json=as_json)

def get_word_portmanteaus(word, as_json=False):
    "Wrapper for get(func='portmanteaus')"
    return get(word, 'portmanteaus', as_json=as_json)

#--- SELF-TEST --------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    import sys

    argp = argparse.ArgumentParser(description='Access RhymeBrain API')
    argp.add_argument('word', type=str, help='word to look up')
    argp.add_argument('outfile', nargs='?', type=argparse.FileType('w'),
                      default=sys.stdout)
    argp.add_argument('-f','--func', type=str, default='rhymes',
                      help='Function to use: rhymes, info, portmanteaus')
    argp.add_argument('-s','--score', type=int, default=0,
                      help='Minimum rhyme score (max 300)')
    argp.add_argument('-n','--num', type=int, default=0,
                      help='Maximum number of results (0 is all results)')
    argp.add_argument('-w','--words', type=int, action='count', default=0,
                      help='output words only')
    args = argp.parse_args()

    word = args.word
    func = args.func
    score = args.score
    num = args.num or None
    words_only = args.words

    result = get(word, func, num, score, bool(words_only), as_json=True)
    args.outfile.write(str(result))
