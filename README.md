# General Assembly Data Science Capstone project

Repo for the capstone project for GA DSI 725

## Project Goal (Problem Statement)

Develop a text generation model that will imitate Shakespeare, and develop
a classification metric to determine the extent that any text is Shakespearian,
and use that metric to evaluate the model.

## Summary

Text processing is one of the many applications for neural networks. Most
recently, transformer models have taken on the role of analyzing and processing
text, with applications in the fields of text translation, text generation, and
text classification. This project applies such models to generate and classify
text based on the works of William Shakespeare.

A transformer neural network is a self-referencing network. I like to think of
it as "time series for non-time sequential data".
So as relates to text, a transformer model will pay attention to past tokens
(words, letters, sentences) to determine future tokens. Such models are deemed
"causal", as they use past tokens to determine future ones, but do not look
into the future.
In more sophisticated transformer models (e.g. masked language models) the
models can look into the future as well as into the past. Such sophisticated
models are used in text translation or sentence completion, where a word in
a sentence needs to be determined based on surrounding words both before and
after that word.
In my case, I'm using both versions. The causal model (GPT2) is used to generate text,
and the masked language model (BERT) is used for classification.

It was previously a goal of mine to develop a model to generate Shakespearean
verse, but this turned out to be a bit overly ambitious.
It turns out some serious academic research has been done on poetry generation
([check out the many papers written on the subject](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C33&q=neural+network+poetry&btnG=)).
The added complexity is immense, as it requires additional non-trivial
consideration of rhyme and meter.
It has therefore become my humble intent to generate text that merely sounds
vaguely Shakespearean (ignoring rhyme and meter), and a scoring system to
score text based on it's "Shakespeareanness". I figure that, with the power
of modern transformer-based text-classification algorithms, the computer
should be able to figure that out on its own with the right input.
And I think, on a rudimentary level, I have accomplished this here.

Text generation output is hard to evaluate, hence my decision to incorporate
a classification model here. To be honest, I just enjoy the ideation of
computer-aided-improvisation, so I would not be so picky to judge
a computer-generated text so long as it vaguely fit my criteria, which in this
case was "Shakespearean". Most of my output did so, so I was happy with the
text generator. However, in the interest of definable metrics, I have
incorporated a classification model which has been trained on various works of
a similar style (and a few of a very un-similar style) which I can use to
classify and score the output.

## Data

See `data` folder.

### Shakespeare Texts

- Shakespeare's complete sonnets, from [Gutenberg.org](https://gutenberg.org/)
- Text from Shakespeare's complete plays, from [Folger's Shakespeare
API](https://folgerdigitaltexts.org/api)

### Other Texts

As needed for classification model, from [Gutenberg.org](https://gutenberg.org/) and procedurally generated via base GPT2 content. Non-procedurally-generated content
was chosen for its stylistic similarity. Authors:

- Oscar Wilde
- William Blake
- John Milton
- Anna Seward
- Percy Shelley
- John Keats
- Hilaire Belloc
- John Donne
- Michael Drayton
- Eleanor Farjeon
- Elizabeth Browning
- Samuel Daniel
- Henry Constable
- Thomas Lodge
- Giles Fletcher
- Bartholomew Griffin
- William Smith
- Robert Lovell
- Robert Southey

## Libraries and Packages

Third-party Python libraries used:

- [`transformers`](https://huggingface.co/docs/transformers/index)
- [`numpy`](https://numpy.org/)
- [`tensorflow`](https://www.tensorflow.org/)
- [`datasets`](https://huggingface.co/docs/datasets/index)

## Process

### Text Preprocessing

Cleaning partially automated to remove any preambles, licenses, and other text
unrelated to content of work (see `cleaning.py` script). Other cleaning manually
performed to tidy up quirks of syntax:

- quotation marks removed; done to mitigate text-generator confusion and
support more fluid text generation

- headings removed, i.e. sonnet numbers; not relevant to text content

- special characters removed, i.e. brackets: not relevant to text content,
often indicating non-content information (such as footnote reference)

- all explanatory text removed, including character names, stage directions,
footnotes, other notes (whether by author or otherwise), etc.;
I considered these not relevant to the content

### Modeling

Used OpenAI's GPT2 pre-trained transformer model as implemented by Hugging
Face's `transformers` library for text generation. Fit on
sentences from all Shakespeare's works, as delimited by ``[.!?:;]``.

Classification model used a BERT attention transformer model as implemented by
Hugging Face's `transformers` library for sequence
classification. Fit on sentences from Shakespeare and other authors' works,
labelled to identify which sentences were from Shakespeare's works, and which
were from other works. The fit BERT model could then classify arbitrary text
into "Shakespearean" or "Not Shakespearean" categories.

These models were then used in tandem, with the classification model scoring
the text generation model. A rough evaluation could therefore be performed on
the text generation to see how often it succeeded in generating what the
classifier considered "Shakespearean" text.
