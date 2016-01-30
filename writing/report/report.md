---
title:  State of the model
author: Fred Callaway
date: \today
---



## Generalization

The goal of the generalization scheme is to allow the model to chunk two words or chunks together even if it has not seen that exact pair before. This is an alternative to traditional approaches to generalization that employ explicit categories. The idea is to allow two tokens to form a chunk if each token can chunk with words that are similar to the other one. For example, if "that cat" has not been seen before, but "that" can chunk with "dog" and "table", words that are similar to "cat", then perhaps "that cat" should also form a chunk.

Our approach `neighbor_generalize` employs explicit links between each node and all other node it forms chunks with. Given "that cat" we create two sets of words: $A$ consists of every node that is chunked with "cat" (e.g. "the") and $B$ consists of every node that "that" is chunked with (e.g. "boy"). We then find all stored chunks from the cartesian product: $A \times B$ (e.g. "the boy"). This is as far as we got at the blackboard.

It's unclear exactly what to do with this set of chunks. Intuitively, the more chunks there are and the more chunky these chunks are, the more likely it is that our initial pair could form a chunk. Currently, we capture this by simply taking the sum of the chunkinesses of all these chunks. In retrospect, this is a bad strategy. (Who knows what I was thinking?) Perhaps a better strategy would be to use the average chunkiness of all pairs in the cartesian product.

However, I am not optimistic about the strategy more generally. This is because it depends on the two potential children ("that" and "cat") already being nodes in the graph. (Only nodes already in the graph could be a member of a chunk in the graph). This is problematic because it prevents generalization over large chunks, which will rarely be added to the graph (see below). Because the chunk created by generalization is by definition not already in the graph, generalization cannot work recursively.

Perhaps our generalization scheme could reflect traditional rule based approaches. Replacing POS labels with vectors, NP $\rightarrow$ D N could be reformulated as "a determiner-like vector combines with a noun-like vector to produce an NP-like vector". It's unclear how to implement such a strategy however. One could not simply bind the semantic vectors of each input vector because the resulting vector would not tell us anything about the context in which NP's occur.

## Chunk ID vector composition

There is a second form of generalization that arises naturally from the creation of chunk id vectors. A chunks id vector is the sum of permuted forms of its children's id vectors. Thus, the chunks `[the dog]` and `[the cat]` will have somewhat similar id vectors because both include a permuted form of `the`'s id vector.

We see in figure 1 that simulations do not provide evidence for the efficacy of this form of generalization. Disabling the composition of id vectors (giving each new chunk a random id vector) has little effect on model performance on the BLEU task.

![BLEU performance with and without composition of id vectors. Six models were trained on 5000 utterances each, three using composed chunk id vectors and three using random chunk id vectors.](id_vector_composition.png)


## Chunk sizes

For a model trained on 5000 utterances in the syllable child directed speech corpus:

Chunk size | number
---|------
2  |  402
3  |   66
4  |   29
8  |    1
6  |    1