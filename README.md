# ScriptWriter
Neural network that fills in blank spaces in movie scripts

## Data:
### Extraction:
1. Website -> HTML ✅ (only about 5% loss)
2. HTML -> script in txt format ✅ (also some % of loss)

## Pre-processing:
1. txt format -> clean txt format (removing useless tabs, repeated \n etc. as little format noise as possible)
2. clean txt format -> tokenized arrays (BPE tokenization but somehow we cant skip \n and \t format is important)

## Models:

1. Contrastive
input: context, fragment
output: similarity score

2. Generative
input: masked context (mask where fragment should start)
output: context + fragment or context + new word + shifted mask

Notes:
- Can't really use fully pretrained transformers since diff vocab because of formatting
- Use some pretrained layers from good transformers

## Idea:

Contrastive model = value net

Generative model = policy net

Combine them with MCTS and generate many path's that the fragment could take


## TODO:

1. DATA TASKS:
  a. Data extraction pipeline improve efficiency !
  b. Data cleaning !!

2. TOKENIZATION !!!
  - How to tokenize data
  - How to include whitespace
  - In general figure out tokenization step
  - Do we need new vocab or can we just use one

## Useful papers for this project:

1. CLIP: https://arxiv.org/pdf/2103.00020.pdf

2. Transformers: https://arxiv.org/pdf/1706.03762.pdf

3. LayerNorm: https://arxiv.org/pdf/1607.06450.pdf
