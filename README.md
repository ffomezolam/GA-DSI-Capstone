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

- 2022-10-10: I'm having trouble getting grips with the underlying workings of
the transformers model, so I've spent most of today and a good chunk of the
weekend reviewing documentation, tutorials, and any articles I can find in the
hopes that demystifying the model would allow me to do more with it.

  Progress is very slow and incremental. For example, it took me all day today
  to finally find out how to stop getting repeating text generating from the
  same prompt - no documentation told me I should be exporting 'tf' tensors in
  order to use sampling, but all the tutorials showed tokenizer returning 'np'
  tensors. Annoying... So it's basically trial and error in order to learn how
  this works. Still, I'm glad that after what seemed like a day of pointless
  busywork, I've made one small improvement.

  So in conclusion, I've got a model that seems to be sensitive to the training
  input and generates text that is somewhat imitative of the training input,
  which has been my main goal. However, I am no closer to poetic verse, and
  I think the poem idea has to be abandoned, as handling rhymes seems to be
  not trivial. **Or** I lean in to the rhyming aspect and try to build
  something that rhymes, and that's the product. As much as I'd love to do
  that, the effort required is sure to sap all of the time available and I am
  not confident that I could have anything working by deadline.

- 2022-10-11: Happened upon some other resources that are relevant to what I'm
doing - poetry and rhyme. Unfortunately, they are both too high-level and too
vague - some of them don't even provide examples of generated results. However
based on these resources I have an idea on how to move forward, and am going to
try a few options tomorrow, starting with an alternative model.

- 2022-10-12: A few things today:

  1. Trying to fit a text generating model on a larger dataset. Very slow on
  my computer. Tried AWS with TensorFlow GPU but GPU didn't engage, and ran
  out of memory. Sigh...

  2. Decided for now to limit to Shakespeare only, because I decided on
     a classification metric - Shakespearian or not? If I have time I'll expand
     to more Elizabethan-era authors. Also determined a perplexity metric could
     work.

  3. Looked up possible classification models. Due to the complications of
     transformers models, I'm probably going to go with one of the BERT models.
     I've considered rolling my own in TensorFlow but it looks like there may
     be far too much to learn to get there within the next few days (would
     probably require me to start subclassing Keras layers etc. which I'm not
     ready to dive into).

  In light of time constraints, my probably-doable modeling plan is:

  1. Get as strong of a text-generation model as I can via larger pre-trained
     sets and longer fit runs.
  2. Build a classification model that will give me an "is Shakespearian"
     score.

  Overall not close to what I was hoping in terms of success but it'll do for
  what I'd consider the minimum requirements at least.
