




- The chunkability of two nodes should depend on how close the composition of the two nodes are to existing nodes.
- This requires searching for nearest neighbors in a high dimensional space which appears to be an unsolved problem.
    - could use 
- composition function could be convolution or multiplication
- Model should learn the chunk threshold

# Word order

- currently we allow two types of depencies: word before and word after. We could account for free word order languages by allowing dependencies between non-adjacent units

# Steps of chunking

for each pair in memory
    get chunkability
    chunk the most chunkable pair
