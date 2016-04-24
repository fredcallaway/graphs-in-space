# Ideas
- how to learn negative information (that nodes don't occur together, despite our expectation that they would)
    - any time a node is seen, add its id_vector to total_id_vector
    - any time a node is seen, subtract .001 * total_id_vector from its row_vector
    - nodes that occur frequently will hold a significant role in total_id_vector, so weights to these nodes will decay faster
- weight not by raw similarity but by a modulated similarity (perhaps make it even more analagous to gravity?)
    - don't normalize row vectors, allowing row vectors that have more information to exert more force


# Shimon Qs
- formalism
    - necessary for BindGraph?
    - mapping from all possible sequences of nodes
    - node has to contain edges
- sims
    - should we use same # utterances for all language, or use larger corpora when we can?


# Compositionality Simulation
- semantic concepts

- dynamic assocation

# TODO
- similarity should be multiplciation of FTP and BTP not addition
- make score function more similar for the two parses
    - maybe using the `within` chunkinesses only helped because it increased the power of the geometric mean sqrt. we should just use utterance len.

- keep all most recently closed chunks in memory, learn at every level

- try to improve NGram models
- finish new numila alogirithm
- trying alternative speaking algorithm
- get fucking generalization to work

# Jones meeting
- why the $\phi$   the `foo`

# Edelman
- tensor similarity
- thesis
    - mention fodor
    - mentions tensors
    - write up 
robert hansen word grammar
    frowned upon by linguists
    words as nodes
    multiple link types

graph is once removed from implementation
    allows link types
    dynamic links?

neural motivation
    vectors are syntactic weights
    granger essential circuits
    smolensky
    inner product between vectors is a fundamental operation of neural network
    the computational brain churchlang sejnowski

merge
    encyclopedia

modularity
    specify upfront

composition and generalization go with main model description

hostel inside old city
print conference paper, schedule for border

## Dynamic generalization
- WHEN bump edge from A to X
    - add A's row_vec to X's dynamic_vec
    - add X's dynamic_vec to A's row_vec
- WHEN bump edge from B to X
    - add X's dynamic_vec to B's row_vec
        - implicitly because dynamic_vec is in id_vec
        - indirectly add A's row_vec to B's row_vec
- what is effect on edge wight revocery?

## Model
- binding operation
    - desiderata
        - a function: V x V -> V
        - non-commutative
        - function depends on both input vectors
    - simple graph surgery
        - C -> A -> B -> D   becomes   C -> AB -> D
        - try de-weighting the bind edges?
    - convolve and map
        - comp_vec = convolve(v1, v2)
        - sem_vec = map(comp_vec)

[foo] 
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
- Numila has a lot of ad hoc stuff
    - "Within-slot interchangeability within a short time window"
- Reading time data for evaluating broad-coverage models of English sentence processing
- Cowan (2000) for choice of 4 window size

- competitive queeuing
    - competition among chunks



- should generalization occur after trying normal chunkiness?
- how to combine the two measures

GEN 0.3 EXEMP 0.3
[numila]  Trained on 1105 utterances in 34605.541867017746 seconds
0.626
{'precision': 0.3732193732193732, 'f_score': 0.49442763824314012, 'recall': 0.655}


GEN 0.4
[numila]  Trained on 1115 utterances in 37820.506844997406 seconds
0.63425
{'recall': 0.735, 'precision': 0.3475177304964539, 'f_score': 0.50539641066680874}