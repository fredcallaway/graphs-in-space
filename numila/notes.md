# Todo
- chunking difficulty measure
- chunk from bag of words
- corpus length effect?
- chunking generalization

# Decisions
- composition operator
    - convolution
    - xOR
- order marking
    - permutations (as in BEAGLE)
    - context vectors
        -  `[the dog]` is `the.after_context + dog.before_contex`?
        - not useful if representing composition as convolution
        - this is wrong bc it makes `[the dog]` like things that preceed `the`
- chunkability
    - FTP and BTP as approximated by context-semantic similarities
    - whether the node is a chunk or near other chunks
- semantic transfer
    - i.e. context vector is part semantic and part id
- decay
    - decay all nodes equally or decay frequently used nodes more?
    - decay with id vector or random vector?
- activations
    - how to model them?
    - how to use them?


# Composition
- The chunkability of two nodes should depend on how close the composition of the two nodes are to existing nodes.
- composition function could be convolution or multiplication
- Model should learn the chunk threshold

## Semantic transfer
- the context vector of a node depends on its semantic vector as well as its id vector

## Word order

- currently we allow two types of depencies: word before and word after. We could account for free word order languages by allowing dependencies between non-adjacent units

## Steps of chunking

for each pair in memory
    get chunkability
    chunk the most chunkable pair


# Decay
If we decay on every step, this will disproportinately affect rare nodes. Perhaps we should decay a small amount on every step, but also every time a node is referred to

# Circular Convolution

## Basics
- circular convolution combines two equal-length verctor to get a same length vector
- circular correlation is the inverse, thus it can be used to reconstruct pieces of the convolved vector __if you know what one of the pieces are.__
    + if $t = c @ x$, then $x ≈ c # t$ 

## For umila
- the goal is to allow chunk-forming rules to generalize categorically.
- one option is to use explicit categories via clustering
- another option is to use implicit categories by using information of similar words to decide about this word

### What we want
- given two vectors representing two words, we can decide how likely they are to bind
    - replaces barlow's principle
- similar pairs have similar bindings i.e. "the dog" and "a cat"
- ideally, we could go the other way, breaking down "the dog" into "the" and "dog"


### Vectors to represent rules
- composition (e.g. of sequences) is probably better represented as addition
- but "the" + "dog" should not be a plain combination of "the" and "dog". "The" and "dog" should combine in a certain way, different from how "the" and "red" would combine.
- need a vector to represent the rule "the" + "dog" -> "the dog"
    - the vector should select "the" and "dog"
    - gives "the" @ D->N + "dog" @ D<-N
    - in the simplest case, you simply have a convolution vector for each branching direction in the tree

### Random vectors?
- perhaps words should have random initial vectors
- "the dog" would just be convolution of the two vectors
- "the __" words will all be somewhat similar, model can learn statistics about them
- two vectors for each word
    + random ID vector used for composition
    + contextual vector used for categorization



At each step we need to decide whether to group the two most recent tokens. Thus we want $p(chunk|t_1, t_2)$


