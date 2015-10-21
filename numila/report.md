---
title:  Umila progress
author: Fred Callaway
date: "October 21, 2015"
---

# Representation
Upon discovery, each node receives an immutable random vector id. This vector is then rotated one position backward and forward to create two context vectors which represent the node being before or after another node: `before_context_vector` and `after_context_vector`. Finally the node receives two edges to every other node in the graph, representing the number of times the node has occured before and after each other node in the working memory window: `backward_edges` and `forward_edges`.

Each is a weighted average of two vectors representing how often the node occurs before and after every other node. This approximately captures backward and forward transitional probabilities (FTP and BTP). Each transitional probability vector is the weighted sum of context vectors for the nodes that have preceded/followed this node.

```python
ftp_vec = sum(node.after_context_vector * (weight / self.count) 
              for node, weight in self.forward_edges.items())
btp_vec = sum(node.before_context_vector * (weight / self.count) 
              for node, weight in self.backward_edges.items())

return (     FTP_PREFERENCE  * ftp_vec + 
        (1 - FTP_PREFERENCE) * btp_vec)
```

# Algorithm
Each utterance is parsed separately. Parsing an utterance consists of passing a 4 node working memory window across the utterance. At each step:

#. Shift a token into memory.
#. Update connection weights between all adjacent nodes in memory.
#. Find the pair in memory with the highest "chunkability". If this pair forms a chunk, replace the pair of nodes with that chunk. Otherwise, shift the least recent node out of memory.

Thus the model can keep more than 4 tokens in working memory by chunking. Additionally, we can see chunking as a primitive precursor to a "merge" function that combines semantic information about two words. With this interpretation, we can define a successful parse as one which can be represented as a single chunk.

Because connection updates occur at each step, the connection strength for every pair of adjacent tokens is increased between 1 and 3 times, depending on when/if a chunk envelops one of the tokens. This means that highly chunkable pairs will receive fewer weight updates because they will often be chunked as soon as they are found.

## Chunkability
The chunkability of two nodes is defined as a weighted sum of cosine similarities between semantic vectors and context vectors. Each cosine similarity serves as an approximation of FTP or BTP. For FTP, we measure similarity between the first node's `semantic_vector` and the second nodes `after_contex_vector`. If the second node has occurred frequently after the first, then the second node's `after_contex_vector` will be a significant part of the sum of context vectors which define the first node's semantic vector. BTP is approximated in the same way, but with the `before_context_vector` of the first node and the `semantic_vector` of the second node.

```python
ftp = cosine_similarity(node1.semantic_vector, node2.after_context_vector)
btp = cosine_similarity(node1.before_context_vector, node2.semantic_vector)
return (FTP_PREFERENCE * ftp + (1 - FTP_PREFERENCE) * btp) / 2
```
