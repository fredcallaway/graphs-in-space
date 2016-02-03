# TODO

## Model
- binding operation
    - C -> A -> B -> D   becomes   C -> AB -> D
    - try de-weighting the bind edges?
- learning at multiple levels
    - build into ID vectors of chunks?
    - do explicitly?
- reconsider learning algo
    - current has less learning for short utterances
    - first 4 always get bumped, not true for others
- phonoloop
    - simulate with activation
    - allow chunks to stick around for short periods of time
- generalization
    - try complete on comprehension task
    - neighbor not 0 - 1
- decay
    - greatly reduces probgraph performance
    - slightly reduced holograph performance

## Analysis
- grammaticality measure
    - chunkedness for all pairs not just actual chunks..
- complete scrambling
- perplexity?

## Abstract
- "we used a modified BLEU score"
- vague, but clear that it's been implemented


# MISC

- Reading time data for evaluating broad-coverage models of English sentence processing
- Cowan (2000) for choice of 4 window size

- competitive queeuing
    - competition among chunks



- should generalization occur after trying normal chunkiness?
- how to combine the two measures