# General Assembly Data Science Immersive Capstone project

Repository for the capstone project for GA Data Science Immersive 725
(concluding 2022-10-17).
This project concerns text generation and classification with neural networks.

## Project Goal (Problem Statement)

Develop a text-generation model that will imitate Shakespeare, develop
a classification model to determine the extent that any text is Shakespearian,
and use the classification model metrics to evaluate the text-generation model.

## Summary

Text processing is one of the many applications for neural networks. Most
recently, transformer models have taken on the role of analyzing and processing
text, with applications in the fields of text translation, text generation, and
text classification. This project applies such models to generate and classify
text based on the works of William Shakespeare.

A transformer neural network is a self-referencing network. I like to think of
it as "time series for non-temporal sequential data".
So as relates to text, a transformer model can pay attention to past tokens
(words, letters, sentences) to determine future tokens. Such models are deemed
"causal", as they use past tokens to determine future ones, but do not look
into the future.
In other, more sophisticated transformer models (e.g. masked language models), the
models can look into the future as well as into the past (to infer based on
surrounding context).
Such sophisticated models are used in text translation or sentence completion,
where a word in a sentence needs to be determined based on surrounding words
both before and after that word.
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
a classification model.
A metric exists, called the "perplexity" metric, to evaluate text-generation
models, but due to compatibility and dependency issues I did not implement one
here.
To be honest, I just enjoy the ideation of computer-aided-improvisation, so I
would not be so picky to judge a computer-generated text so long as it vaguely
fit my criteria, which in this case was "Shakespearean".
Most of my output did so, so I was happy with the text generator.
However, in the interest of definable metrics, I have incorporated a
classification model which has been trained on various works of
a similar style (and a few of a very un-similar style) which I can use to
classify and score the text-generation output.

The resulting classification model scored with 92% accuracy on
predicting the classes of a test data set, which was stratified to match the
label proportion on the entire set.
As a reference point, the Shakespeare data set was about 67% of the entire data
set, so a naive baseline accuracy would be 67%.
This model therefore scored significantly higher, indicating some success in
discerning Shakespeare from other authors.

Finally, I generated text from three test datasets, which were comprised of
sentence fragments from various texts:
1. Wine descriptions from [a Kaggle-hosted
   dataset](https://www.kaggle.com/datasets/zynicide/wine-reviews)
2. Shakespeare's complete works (the positive class data for the classification
   model)
3. Other words (the negative class data for the classification model)

100 samples from each set were generated and classified, and scores were
obtained from each classification set:

1. Wine description fragments were 68% Shakespearean, with a mean probability
   score of 0.62.
2. Shakespeare fragments were 96% Shakespearean, with a mean probability score
   of 0.89.
3. Other fragments were 49% Shakespearean, with a mean probability score of
   0.49.

These scores indicate that Shakespearean prompts will yield more Shakespearian
results, but also that on unrelated prompts (e.g. wine descriptions) the
results can be Shakespearean more often than not. More tests could be done here
to experiment with sentence fragment size and type to determine what sorts of
input yielded the best output.

Bespoke tests of the generation-classification pair (i.e. arbitrary text input
fed into the classifier) suggested that prompts
which sounded more Shakespearean resulted in higher Shakespearean scores, whereas
prompts that included modern terminology or phrasing tended towards lower
scores. There is no obvious way to interpret this *vis-a-vis* the model. Is the
model overfit if it doesn't consider non-Shakespearean language to be
Shakespearean? Is the model underfit if it categorizes Shakespearean language
following a non-Shakespearean prompt as not Shakespearean? It would seem this
is all depended on the downstream goal. But I was just looking to generate some
text that sounds like Shakespeare, and this did so pretty alright! So I'll
score it 0.92!

## Directory Structure and Index

- `code/`: scripts, notebooks, and other code
- `models/`: where models should be kept for default configuration
- `data/`: text files for model training

## Data

See `data/` folder.

All text files are included in the repository. For causal model validation, please
download [the Wine Reviews datasets from
Kaggle](https://www.kaggle.com/datasets/zynicide/wine-reviews) and move the
`winemag-data-130k-v2.csv` file into this directory.

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
- [`scikit-learn`](https://scikit-learn.org/stable/)
- [`pandas`](https://pandas.pydata.org/)
- [`streamlit`](https://streamlit.io/) (if looking to run streamlit app locally)

## Process

### Text Preprocessing

See `code/utilities/cleaner.py`

Cleaning was partially automated to remove any preambles, licenses, and other text
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

See `models/` directory for instructions on downloading and setting up
pre-trained models

Used OpenAI's GPT2 pre-trained transformer model (`distilgpt2`) as implemented by Hugging
Face's `transformers` library for text generation. Fit on
sentences from all Shakespeare's works, as delimited by ``[.!?:;]``.

Classification model used a BERT attention transformer model
(`distilbert-base-uncased`) as implemented by
Hugging Face's `transformers` library for sequence
classification. Fit on sentences from Shakespeare and other authors' works,
labelled to identify which sentences were from Shakespeare's works, and which
were from other works. The fit BERT model could then classify arbitrary text
into "Shakespearean" or "Not Shakespearean" categories.

### Evaluation

The classification model was evaluated using standard binary classification
metrics:
- Baseline was approx 67% accuracy, as that was the relative size of the
positive class compared to all data.
- Classification accuracy was 92%. A significant improvement over baseline.

In order to evaluate the causal (text generating) model, I had the classification
model score the text generation model on various test input fragments.
A rough evaluation could therefore be performed on
the text generation to see how often it succeeded in generating what the
classifier considered "Shakespearean" text.

Three sets were provided to the generator, and passed along to the classifier:
1. Sentence fragments from wine descriptions from [a Kaggle-hosted
   dataset](https://www.kaggle.com/datasets/zynicide/wine-reviews).
2. Sentence fragments from Shakespeare's works (the positive class data)
3. Sentence fragments from other works (the negative class data)
The sentence fragments were short portions of sentences from each set, which
were fed to the causal model to generate data, which was limited to only the
first sentence of that data. This was fed to the classifier.

The classification results showed the highest positive (Shakespearean)
classifications for sentence fragments sourced from Shakespeare's works. The
other two categories (wines and other) showed 68% and 49% Shakespearean
results, respectively.

So, as probably should be expected, Shakespearean in tends to Shakespearean
out. But the reasonably high success rate in the other categories would suggest
that the causal model can generate reasonably Shakespearean text from arbitrary
input.

## Conclusion

This project really just scrapes the tip of the iceberg of language generation
modeling. One of the major defining aspects of Shakespearean style that is lost here is the
verse - the rhyme scheme and poetic meter that dominates his works. Poetic
rhyme and meter is very hard to model, as evidenced by the many academic papers
which struggle with the attempts. So, while the word choice and structure may
lean towards the Shakespearean in this data model, the real essence of what
makes Shakespeare (and poetry in general) great is missing entirely. But enough
of the highbrow chatter, let's generate some "thee"s and "thou"s!
