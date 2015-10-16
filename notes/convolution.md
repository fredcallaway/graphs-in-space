---
title: Convolution
date: 2015-10-02 17:13:26
tags: umila
tags:
---

# Basics
- circular convolution combines two equal-length verctor to get a same length vector
- circular correlation is the inverse, thus it can be used to reconstruct pieces of the convolved vector __if you know what one of the pieces are.__
    + if $t = c @ x$, then $x ≈ c # t$ 

# For umila
- the goal is to allow chunk-forming rules to generalize categorically.
- one option is to use explicit categories via clustering
- another option is to use implicit categories by using information of similar words to decide about this word

## What we want
- given two vectors representing two words, we can decide how likely they are to bind
    - replaces barlow's principle
- similar pairs have similar bindings i.e. "the dog" and "a cat"
- ideally, we could go the other way, breaking down "the dog" into "the" and "dog"


## Vectors to represent rules
- composition (e.g. of sequences) is probably better represented as addition
- but "the" + "dog" should not be a plain combination of "the" and "dog". "The" and "dog" should combine in a certain way, different from how "the" and "red" would combine.
- need a vector to represent the rule "the" + "dog" -> "the dog"
    - the vector should select "the" and "dog"
    - gives "the" @ D->N + "dog" @ D<-N
    - in the simplest case, you simply have a convolution vector for each branching direction in the tree

## Random vectors?
- perhaps words should have random initial vectors
- "the dog" would just be convolution of the two vectors
- "the __" words will all be somewhat similar, model can learn statistics about them
- two vectors for each word
    + random ID vector used for composition
    + contextual vector used for categorization
