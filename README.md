# General Assembly Data Science Capstone project

Repo for the capstone project for GA DSI 725

In progress.

## Project Goal (Problem Statement)

Develop an AI model that can learn and imitate a writing style. Possible reach
or limiting goals:

- Limit scope to poetry, or a particular form of poetry, i.e. sonnet
  - Probably will need to use some form of transformer model to generate text
  - Problem: how to handle poetic metre?
- Generalize by POS tagging
  - Can possibly train a model based on POS tags
  - Problem: generating speech from tags may be more difficult than generating
      from learned prediction

## Progress and Findings

- 2022-10-06: Exploring possible models for generating sentences, currently
HuggingFace GPT2, which wants to throw errors as often as possible. Have been
able to generate conclusions to lines from Shakespeare's Sonnets, but these are
not very promising so far, with only approx 2/10 generating usable phrases

  I have also been working on a word generator based off a custom trie
implementation of something like a markov chain, but developing what is
essentially a basic model from scratch seems like it may take too long to be
feasible within the deadline. The hope was to combine generated words with POS
tags to emulate poetry styles using fake words, resulting in something like
Carroll's Jabberwocky.

- 2022-10-07: End of work day progress report. After struggling for all
yesterday and a good chunk of today with the transformers library, I think I'm
finally making some progress, mostly thanks to some very hard to find write-ups
on its usage. Unfortunately the documentation and tutorials do not give very
clear explanation about what things do, so I've been forced to piece it
together from code snippets and often ambiguously-worded write-ups and
code-alongs.
  
  I've gotten the model to start spitting out decent sounding text that doesn't
end up in an endless loop. So that's good. Unfortunately, looks like I have to
go back to cleaning and possibly sourcing additional text to feed the model:

1. Additional cleaning needs to be done to handle weird apostraphes, quotes,
   underscores, and anything else that might mess up the tokenizer. I'm going
   maximum brute force here.
2. I feel like I need to feed more text to the model. Not worried about getting
   the text - I'm more worried about speed of fitting the model. If necessary
   I'll grab AWS for it. More to come over the weekend!

Also, I think I'm limiting my project scope to just generating sonnet-like text
from a prompt or line. We'll see if that sticks.
