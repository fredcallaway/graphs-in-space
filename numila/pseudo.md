---
title:  Pseudocode
author: Fred Callaway
date: "November 11, 2015"
---


The learning process consists of parsing a list of utterances, one by one.
Each utterance is parsed separately using the knowledge gleaned from earlier utterances. In addition to outputting a bracketed parse for an utterance, parsing an utterance causes learning. In this way, the model exemplefies a ussage-based approach to language acquisition.

## Main loop
The following is the main loop in `Parse`. Details for each step are given below

```python
# self is a list of nodes
# At the end of the process, this list reprsents a parse tree
for token in utterance:
    self.graph.decay()                                 # 1. Decay
    self.shift(token)                                  # 2. Shift
    self.update_weights(self.params['MEMORY_SIZE'])    # 3. Update weights
    if len(self) >= self.params['MEMORY_SIZE']:        # 4. Chunk
        # fill memory before trying to chunk
        self.try_to_chunk(self.params['MEMORY_SIZE'])  
```

1. __Decay:__ All edge weights in the graph decay.

2. __Shift:__ A new token is added to memory. If there are already `MEMORY_SIZE` (default 4) nodes in memory, then the least recent one falls out of memory. It will be included in the final parse, but no more processing is done on it.

3. __Update weights:__ For each pair of adjacent nodes in memory, $n_1\ n_2$, (1) increase the weight of the forward edge from $n_1$ to $n_2$ and (2) increase the weight of the backward edge from $n_2$ to $n_1$.

4. __Chunk:__ The _chunkability_ of each pair of adjacent nodes is a weighted average of the forward edge weight from $n_1$ to $n_2$ and the backward edge weight from $n_2$ to $n_1$. If the pair with the highest chunkability forms a chunk, the two consituent nodes are removed from memory and replaced with the chunk. The model does not attempt to form chunks unless it has a full memory OR there are no tokens left in the incoming utterance (code not shown).

## Holographic graph reprsentation
Numila represents a graph in a distributed way using holographic representations. The traditional N x N adjacency matrix is replaced by an N x D matrix where D is the dimension of our sparse vectors. Each row represents the outgoing edges of a node as the summation of the id-vector for every node it is connected to. We weight this sum to get weighted edges. To represent multiple edge types, we use permutations. Each edge type is assigned a random permutation vector, an _edge-vector_. To update a specific edge from node $n_1$ to $n_2$, we add the id-vector of $n_2$ permuted by the corresponding edge-vector to the row of $n_1$. Thus we can define the row for a node $n_0$ as

$$ \text{row-vector}(n_0) = \sum_{e\in E} 
    \sum_{n \in N} \text{permute}(\text{id-vector}(n), \text{edge-vector}(e))
                   \cdot \text{edge}(e, n_0, n) $$

where $E$ is the set of edge types, $N$ is the set of nodes, $\text{permute}(x, y)$ permutes vector $x$ by vector $y$, $\text{id-vector}(n)$ returns the id-vector of node $n$, and $\text{edge}(e, n_1, n_2)$ returns the weight of the edge of type $e$ connecting $n_1$ to $n_2$. Note that we assume all possible edges exist with default 0 weights.

To recover edge weights, we take the cosine similarity of a node's row and the permuted vector of another node. Intuitively, this value will be higher if the second node's permuted id-vector is a large part of the sum that defines the first node's row-vector.

For an edge type $e$ connecting $n_1$ to $n_2$, we define the edge weight as

$$ \cos (\text{row-vector}(n_1), \text{permute}(e, \text{id-vector}(n_2))) $$

