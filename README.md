# ScriptWriter
Neural network that fills in blank spaces in movie scripts

## Data:
1. Website -> HTML (Mostly done ✅, only about 5% loss)
2. HTML -> unified format (Not unified but done ✅)
3. unified format -> tokenized arrays 

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

1. TOKENIZATION



## Useful papers for this project:

1. CLIP: https://arxiv.org/pdf/2103.00020.pdf

2. Transformers: https://arxiv.org/pdf/1706.03762.pdf

3. LayerNorm: https://arxiv.org/pdf/1607.06450.pdf
