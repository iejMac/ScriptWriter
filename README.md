# ScriptWriter
Neural network that fills in blank spaces in movie scripts

## Data:
1. Website -> HTML (Mostly done ✅, only about 5% loss)
2. HTML -> unified format
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
