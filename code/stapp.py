import streamlit as st

from utilities.utilities import load_config
from utilities.utilities import get_model_path, load_tokenizer, load_model
from utilities.utilities import generate_from, classify_from
from utilities.utilities import extract_sentences

import random

# quotes referenced from memory and from goodreads.com
quotes = {
    "neutral": [
        "Speak the speech, I pray you, as I pronounc'd it to you, trippingly on the tongue",
        "Et tu, Brute?",
        "Perdition catch my soul",
        "For a charm of powerful trouble, like a hell-broth boil and bubble",
        "Some are born great, others achieve greatness",
        "Better a witty fool, than a foolish wit",
        "Sweets to the sweet"
    ],
    "positive": [
        "Ay, there's the rub",
        "The apparrel oft proclaims the man",
        "With mirth and laughter let old wrinkles come",
        "Shall I compare thee to a summer's day?",
        "All's well that ends well",
        "How noble in reason! how infinite in faculty!",
        "The beauty of the world! the paragon of animals!",
        "Good name in man and woman, my dear lord, is the immediate jewel of their souls"
    ],
    "negative": [
        "You speak an infinite deal of nothing",
        "In this sleep of death what dreams may come...",
        "Conscience doth make cowards of us all",
        "Wisely and slow; they stumble that run fast",
        "Better a witty fool, than a foolish wit",
        "Some Cupid kills with arrows, some with traps",
        "Thine face is not worth sunburning",
        "Double, double toil and trouble",
        "Thou mad mustachio purple-hued maltworms!",
        "Who steals my purse steals trash",
        "Thou art a very ragged Wart"
    ]
}

def get_quote(cat='neutral'):
    qs = quotes[cat]
    qlen = len(qs)
    return qs[random.randint(0,qlen-1)]

st.title("Are You Shakespearean?")
st.write('*' + get_quote('neutral') + '*')

CONFIG_FILE = 'config-test.json'

cfg = load_config(CONFIG_FILE)
causal_model_path = get_model_path(cfg, 'causal')
class_model_path = get_model_path(cfg, 'class')

causal_tokenizer = load_tokenizer(cfg['CAUSAL_MODEL'])
causal_model = load_model(causal_model_path, 'causal')

class_tokenizer = load_tokenizer(cfg['CLASS_MODEL'])
class_model = load_model(class_model_path, 'class')

text = st.text_input("Input")
generate = st.checkbox("Generate follow-up text", value=0)
sentence = 0
if generate:
    sentence = st.checkbox("One sentence out", value=1)

def score_text(text):
    if generate:
        gen = generate_from(text, causal_model, causal_tokenizer)
        if sentence:
            gen = extract_sentences(gen)[0]
    else:
        gen = text

    classification = classify_from([gen], class_model, class_tokenizer)
    classification = classification.get_results()[0]

    st.subheader("Output")
    st.write(classification[0])

    st.subheader("Class")
    c = "Shakespearean" if classification[1] else "Amateurish"
    st.write(f'You are *{c}*')
    quote = get_quote('positive' if classification[1] else 'negative')
    st.write(f'*"{quote}"*')

    st.subheader("Score")
    st.write(str(round((float(classification[2]) * 100), 2)))

if text: score_text(text)
