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
