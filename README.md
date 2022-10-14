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

It was previously a goal of mine to develop a model to generate Shakespearean
verse, but this turned out to be a bit overly ambitious. It turns out some
serious academic research has been done on poetry generation, and it is very
complex and difficult as it requires additional consideration of rhyme and
meter. It is therefore my humble intent to generate text that
merely sounds vaguely Shakespearean (ignoring rhyme and meter), and a scoring system to score text based
on it's "Shakespeareanness". I believe that, on a rudimentary level, this was
accomplished here.

## Data

See `data` folder.

### Shakespeare Texts

- Shakespeare's complete sonnets, from [Gutenberg.org](https://gutenberg.org/)
- Text from Shakespeare's complete plays, from [Folger's Shakespeare
API](https://folgerdigitaltexts.org/api)

### Other Texts

As needed for classification model, from [Gutenberg.org](https://gutenberg.org/) and procedurally generated via base GPT2 content. Authors:

-

## Libraries and Packages

Third-party Python libraries used:

- [`transformers`](https://huggingface.co/docs/transformers/index)
- [`numpy`](https://numpy.org/)
- [`tensorflow`](https://www.tensorflow.org/)
- [`datasets`](https://huggingface.co/docs/datasets/index)

## Process

### Text Preprocessing

Cleaning partially automated to remove any preambles, licenses, and other text
unrelated to content of work. Other cleaning manually performed to tidy up
quirks of syntax:

- quotation marks removed; done to mitigate text-generator confusion and
support more fluid text generation

- headings removed, i.e. sonnet numbers; not relevant to text content

- special characters removed, i.e. brackets: not relevant to text content,
often indicating non-content information (such as footnote reference)

- all explanatory text removed, including character names, stage directions,
footnotes, other notes (whether by author or otherwise), etc.; I considered these not relevant to the content

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
