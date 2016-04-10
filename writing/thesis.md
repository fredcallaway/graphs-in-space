---
title: "Representing graphs with sparse vectors."
author: Fred Callaway
date: \today
---

# Introduction
- learning structure in the environment
- graph as tool to represent structure
- language as a graph
- graph can represent both simple and complex models
- graph as a bridge from simple statistical models to complex linguistic models

# Graphical models of language
The earliest statistical language models, and many of the most commonly used models today [??], are based on transitional probabilities. These models attempt to capture regularities in language by learning the statistical dependencies between adjacent words. The simplest such model is the bigram model, which treats language as a first order Markov process: each word is assumed to depend probabalistically on only the previous word. A bigram model is generally represented as a transitional probability matrix, or equivalently a graph with words as nodes and transition probabilities as edges. In this model, an utterance can be produced by starting at a node $n_0$ (often a special START node), and then choosing the next a node $n_1$ with probability equal to the edge weight from $n_0$ to $n_1$. This process can be iterated until a termination criteria is reached (often the arrival at a special STOP node).

Even under the false assumption that people speak purely based on statistical dependencies between words, the bigram model is fundamentally lacking. Language is rife with long distance dependencies such as "either-or" that a bigram model cannot possibly capture. One strategy to capture long distance dependencies is to increase the order of the Markov process. For example, a second order Markov process, or trigram model, assumes that a word depends on both the previous word and the word before that one. With some squinting, a trigram model can be represented as a standard directed graph with two words in each node. For example, the transitional probability $p(w_i = z | w_{i-1} = y, w_{i-2} = x)$ would be represented as the edge between the node $n_{xy}$ and $n_{yz}$. [keep/expand?]

However, increasing the Markov order has the undesirable side effect of exponentially increasing the dimensionality of the space. There are $n^N$ possible N-grams, where $N$ is the markov order and $n$ is the vocabulary size. Thus, as $N$ increases, the percentage of grammatically valid N-grams that the learner will actually be exposed to will decrease exponentially. Many techniques in Natural Language Processing are designed to get around this problem of data sparsity, such as smoothing or variable order N-grams. For example, the backoff algorithm measures all N-gram probabilities of $N < N_{max}$, and dynamically decides which order, $N$ to use in a probability estimation based on the number of relevant N-grams it has stored for each $N$ [@katz87]. This idea of tracking variable length sequences is a fundamental basis of the present model, and the two models upon which it is based.

The ADIOS model [@solan05] explores one such technique that aims to respect the hierarchical nature of language. Unlike N-gram models which always predict the single next word based on some number of previous words, ADIOS directly models the statistial dependencies between multi-word units, e.g. between "the dog" and "ate the steak". These multi-word units or "patterns" are constructed recursively through an iterative batch-learning algorithm. When two nodes (each of which may represent any number of words) are found to frequently occur adjacently in the corpus, they are combined into a new node. Later iterations may discover that this node occurs frequently with another node, allowing the creation of deep hierarchical patterns. TODO: More?

Although ADIOS demonstrated the utility of graphical representations in language modeling, the batch learning algorithm it employed casts some doubt on its relevance as a psychological model. To address this concerrn @kolodny15 created U-MILA, an incremental model based on ADIOS but intended to more closely reflect human learning abilities. The model is incremental, passing through the corpus a single time, building up the graph from an initially clean slate. TODO: More.

## Graphical models and traditional syntactic theories

Network theories of language often suggest that language is most fundamentally described by relationships between individual words [@hudson03], while traditional linguistic theory emphasizes composition and relationships between constituents [@chomsky65; @chomsky95; @stabler96]. Although these views are frequently seen as opposing, they may serve complimentary roles. Perhaps direct word-word relationships characterize early linguistic knowledge that is only later developed into more complex syntactic structures [@bannard09]. It is also possible that the representations of formal linguistics should be viewed as theoretical abstractions that roughly characterizes a fuzzier, network-based language system [@lamb99]. @marr82 calls Chomsky's (1965) theory of transformational grammar a "true computational theory", suggesting that Chomsky's performance-competence distinction reflects Marr's algorithmic-computational distinction. Under this interpretation, network and phrase structure theories may be posed at different levels of analysis; if the theories make roughly similar predictions, they may not conflicting.


# MergeGraph (sorry)
We describe an augmented graphical datastructure, intended to represent domains that are defined by complex inter-relationships and compositional structure. Language involves more than the relationships between words and constituents [citation]; however, the structural aspects of language may be modeled well with such a datastructure. A MergeGraph is based on a weighted, labeled, directed, multigraph. That is, a node may have multiple edges pointing to another node, representing different types of connections. Additionally, nodes in a MergeGraph may be composed of other nodes, inspired by Higraphs [@harel88]. For example, the constituent `[the dog]` could be a single node, composed of the nodes `the` and `dog`. This node could have two edges to the node `scared` labeled _argument_ and _subject_.




A MergeGraph is a tuple $G = TODO$ where

- $V$ is a finite set of vertices.
- $\Sigma_E$ is a finite set of edge labels.
- $E$ is finite set of edges, each of which is a tuple $(x, y, \ell)$ where $x$ and $y$ are vertices, and $\ell$ is label $\in \Sigma_E$
- $fweight$ is a function where $fweight(x, y, \eps)$ is the edge weight of type $\eps$ from $x$ to $y$
- $fbump$ is a function where $fbump(X, Y, \eps)$ increases the edge weight of type $\eps$ from $X$ to $Y$
- $fmerge$ is a function where, for a list of nodes $\mathbf{x}$, $fmerge(\mathbf{x})$ is a node $y$ where $y$ represents the ordered composition of the nodes in $\mathbf{x}$.


Not sure how to formally describe a mutable datastructure. Note that the node actually contains all its edges, which is what makes merge work.

Like a standard graph, MergeGraph is an abstract data type, meaning it can be implemented in a number of ways. In the case of a MergeGraph, however, the implementation has a profound impact on the behavior.



- modified higraph with ordered merge rule
    - chomsky, hagoort

# Sparse vector graphs
- motivation
    - neural
    - generalization
- background on sparse vectors
- implementation


# Nümila
In order to test the sparse vector graph representation, we created a simple language acquisition model, Nümila, based on previous graphical models [@solan05; @kolodny15]. Like these previous models, the present model reduces the problem of language acquision to the much simpler problem of producing grammatical utterances based on statistical patterns in speech. In reality, language acquisition is heavily dependent on the semantic and social aspects of language [@tomasello03; @goldstein10], aspects which the present model does not capture. However it is generally agreed that linguistic pattern recognition plays at least some role in language acquisition; thus, the present model can be seen as a baseline that could be improved upon by enriching the input to the model with environmental cues. TODO We discuss ways that these cues could be incorporated into a graphical model.

## Graphical model

### Edges
The model has two edge-types representing forward and backward transitional probabilities. Although forward transitional probability (FTP) is the standard in N-gram models, some evidence suggests that infants are more sensitive to BTP [@pelucchi09], and previous language acquisition models have been more successful when employing them [@mccauley11]. To examine the relative contribution of each type of TP, we make their relative weight an adjustable parameter. Although ADIOS and U-MILA have only one type of temporal edge (simply a count of coocurrences), their learning algorithms compute something very similar to FTP and BTP [TODO proof?]. By using two edge types, we build this computation into the representational machinery.


### Merge
When two nodes (initially words) are determined to cooccur at an unexpectedly high frequency (see below), the graph's merge function is applied to create a new node. As discussed above [TODO], the merge function, $fmerge$, is a parameter of the graph, and must be a function from a list of nodes to a single node. As a simplifying assumption, we follow U-MILA by only considering binary merges; thus the function is of type $N \times N \rightarrow N$. Importantly, we do not make the theoretical claim that linguistic merges must be binary [as others have @everaert15]. We implement two such merge functions, one hierarchical and the other flat. Given arguments `[A B]` and `[C D]`, hierarchical merge returns `[[A B] [C D]]`, whereas flat merge returns `[A B C D]`.

In the simplest case, the merge function determines only the identity of the new node. However, this quality alone has important ramifications. Hierarchical merge is a bijective function; that is, there is a one-to-one mapping from inputs to outputs. Conversely, flat merge is not bijective because multiple inputs can produce the same output. For example, if `[A B C]` and `D` occur together frequently, a new node `[A B C D]` will be created and added to the graph. Later on, if `[A B]` and `[C D]` are merged, we will get the existing node, `[A B C D]` with all its learned edge weights. In more practical terms, a model using a flat merge rule will treat every instance of a given string e.g. "Psychology department chair" as the same entity. Although there are clear semantic reasons why a flat merge function would be undesirable, it is not clear to what extent hierarchical information will be useful for the purely grammatical tasks we test our models with TODO.

The full power of the merge function, however, comes from its ability to construct initial edge weights for the new node. This allows truly compositional structure, where the behavior of larger units is predictable based on its constitutents. This is critical for syntactic processing: Even if you had never heard the phrase "honest politician", you could still predict its syntactic behavior. In a graph, the syntactic behavior of a node is represented in its edge weights. Thus, predicting syntactic behavior comes down to constructing an initial edge profile for the newly created node based on the edge profiles of its element nodes. This function can be specified by the modeler or learned. [TODO specified merge function]


## Learning
- high level declarative description of weights?

The model constructs a Higraph given an initially blank slate by processing each utterance, one by one. Thus, the model has more limited memory resources than both ADIOS and U-MILA. The graph is constructed with four operations: (1) adding newly discovered base tokens to the graph, (2) increasing weights between nodes in the graph, (3) measuring weights between nodes, and (4) creating new nodes by merging existing nodes. We implement two processing algorithms that employ these basic operations to both learn from and assign structure to an utterance. The first is meant to replicate U-MILA's bottom up chunking mechanism, learning transitional probabilities between all possible nodes in the utterance. The second is inspired by the Now-or-Never bottleneck [@christiansen15], incorporating an even more severe memory constraint, and building up a structural interpretation of the utterance word by word.

<!-- 
The core of the learning algorithm lies in (1) updating the weights between adjacent pairs nodes, and (2) considering the merging of adjacent pairs of nodes. The first consists of increasing  $E_F(X, Y)$ and $E_B(Y, X)$.
 -->

### FullParse

The FullParse algorithm is similar to U-MILA's bottom up learning algorithm in that it finds all pairs of adjacent nodes (potentially overlapping) in the utterance and then applies a learning process to the pair. The algorithms differ in the edge weights that are updated, and the criteria for creating a chunk. For every pair of nodes $(X, Y)$ such that $X$ directly precedes $Y$ in the utterance:

#. Increase weights
    #. Forward transition probability $E_F(X, Y)$
    #. Backward transition probability $E_B(Y, X)$
#. Attempt to create a chunk
    #. Check that merging the pair would not result in a node already in the graph
    #. Check that the _chunkiness_ of the pair exceeds a fixed threshold, where _chunkiness_ is the geometric mean of the forward and backward transition probabilities between the nodes: $\sqrt{E_F(X, Y) \cdot E_B(Y, X)}$
    #. Create a new node by merging $X$ and $Y$
    #. Add this node to the graph

Unlike U-MILA's algorithm, this algorithm requires the full utterance to be in memory before any processing is done. This is because the specific algorithm isn't meant to be cognitively realistic [TODO should this be discussed, perhaps footnoted?]

### GreedyParse

The GreedyParse algorithm follows the same basic principles as FullParse in that it is based on updating transitional probability edges and merging nodes. However, unlike FullParse, GreedyParse incorporates severe memory limitations and online processing restraints in line with the Now-or-Never bottleneck [@christiansen15]. In contrast to FullParse, which finds all possible structures for an utterance given the current graph, GreedyParse finds a single structure by making a sequence of locally optimal decisions, hence "Greedy". Upon receiving each word it can create at most one chunk and the nodes used in this chunk can not be used later in a different chunk. Thus, the algorithm may not assign the optimal structure to the utterance.

```
# Shift.
append a new token onto memory
if token is not in graph:
    add token to graph
fi
update weights between the previous token and the new token

# Chunk.
select pair of adjacent nodes with maximum chunkiness
if chunkiness of pair exceeds threshold:
    create chunk by merging the two nodes
    if chunk is not in graph:
        add chunk to graph
        update weights between new chunk and adjacent nodes
    fi
else:
    remove the oldest node from memory
fi
```

### Generalization
Thus far, we have described learning algorithms that are essentially bigram models with variable length elements. Thus, like any N-gram model, the these algorithms will suffer from the problem of data sparsity. Incorporating variable-length units increases this problem dramatically because the number of tracked elements is no longer limited to the vocabulary size. To account for this, a su

### Compositionality
Presumably, experience with pairs of similar words informs your expectations for how the syntactic role of this novel phrase. The standard linguistic explanation involves rules and categories: "honest" is an `Adj`; "politician" is a `N`; and thus by the rule `N' -> Adj N`, "honest politican" is an `N'`. The extensive regularity of language implies that such theories are at the very least a useful computational-level theory [as @marr82 suggests]. However, it is an open question whether adult speakers truly represent rules and discrete categories. It's possible that rule-like behavior emerges from a system that explicitly represents only relationships between individual items [citation].

- rules and hard categories not characteristic of brains
- computational vs algorithmic
- how to learn?

perhaps based on your experience with other adjective-noun pairs.  the new nodes edges would be similar to other nodes that were composed of similar elements.

## Grammaticality judgement
A standard measure of a language model's quality is its ability to discriminate grammatical from ungrammatica utterances. To perform this task, a generative language model can assign probabilites of producing each utterance, and then choose a cut off, judging utterances above that probability to be grammatical.

Nümila is not a language model in the formal sense because it does not represent a probability density function over possible utterances [citation]. This is partially a result of the fact that outgoing edge weights for one node are not required to sum to 1, as they are in an N-gram model, which is a language model in the formal sense. However, although Nümila cannot assign a probability to a given utterance, it does have a function that is _proportional_ to a probability density function. Specifically Nümila can assign a numeric score to any given utterance that represents its acceptability on a zero-to-one scale. Recall that each learning algorithm assigns one or more parses to an utterance, where a single parse is a list of non-overlapping nodes, e.g. $(A, (B, C), (D, (E, F)))$. This structure could also be represented as a tree. [TODO tree diagram]

A standard way to assign a probability to a parse tree in linguistics is to take the product of the probabilities of each rule needed to construct the tree. This list of rules is determined by the linguistic model. For example, with a PCFG, the rules are all of the form $NT \rightarrow \alpha$, where $NT$ is a nonterminal symbol (representing one branch of the tree, a constituent) and $\alpha$ is a sequence of symbols, either terminal or nonterminal. With an N-gram model, on the other hand, the rules are all of the form $\alpha \rightarrow \alpha \cdot w$, where $\alpha$ is the $n-1$ most recent words and $w$ is the next word.

Nümila uses both types of rules in its scoring algorithm. For chunks, the PCFG rule is applied. The node is $X$ and its children are $alpha$; because there is only one composition for a given node, the rule probability is always 1. When an utterance has a series of nodes that cannot be combined, we apply a bigram rule. For each adjacent pair of nodes, $(X, Y)$, we apply the rule $X \rightarrow X \cdot Y$ with probability proportional to $chunkiness(X, Y)$. Because all non-terminal generation rules have probability 1, the result is simply the product of chunkinesses between each top level node in the utterance. Finally, to avoid penalizing longer utterances, we take the result to the $n-1$^th root where $n$ is the length of the utterance. Taking the above example, we find

$$score = \sqrt[5]{chunkiness(A, [B, C]) \cdot chunkiness([B, C], [D, [E, F]])}$$

Given a function that assigns scores to a single parse of an utterance, it is straightforward to create a function that assigns scores to the utterance itself. With a PCFG (where the scores are probabilities) the probability of the utterance is the sum of the probabilities of each parse. This is a result of the assumption that the utterance is generated by the PCFG model, along with Kolmogorov's third axiom that the probability of the union of independent events is the sum of the probabilities of each event. Although Nümila's scores are not true probabilities, we apply the same rule. That is, the score of an utterance is the sum of the scores for all parses of that utterance. [TODO does this maintain proportional probability?]

Recall that when an utterance is parsed using the GreedyParse algorithm, only one possible parse is found. Thus, the score of the utterances is simply the score of that single parse. As a consequence, the GreedyParse algorithm prefferentially treats utterances that have a narrower, or lower-entropy, distribution of scores for all possible parses. Conversely, an utterance that has many possible parses, all with somewhat high scores, will be at a disadvantage.

## Production
TODO production for FullParse algorithm and flat merge.

- extras
    - generalization
    - composition
     - reference Baroni

## Results
To test the model, we use naturalistic child directed speech. [TODO should we be using syllables or words, perhaps try both?]. We use corpora prepared by @phillips14, which include syllabified and word-tokenized versions of CDS corpora in 7 languages. We test several instantiations of Nümila using different Graph implementations, parsing algorithms, and merge functions.

### Grammaticality discrimination
As a first test, we use the common task of discriminating grammatical from ungrammatical utterances. To construct a test corpus, we first take # unseen utterances from the corpus, which are labeled "grammatical". For each utterance, we create a set of altered utterances, each with one adjacent pair of tokens swapped. For example, given "the big dog", we create "big the dog" and "the dog big". These altered utterances are added to the test corpus with the label "ungrammatical". The models task is to separate grammatical from ungrammatical. Often, this task is modeled by setting a threshold, all utterances being predicted to be grammatical. However, it is unclear how to set such a threshold without either specifying it arbitrarily or giving the model access to the test labels. Thus, we employ a metric from signal detection theory, the Recevier Operator Characteristic.

The ROC curve plots true positive rate against false positive rate. As the acceptability threshold is lowered, both values will increase. With random scores, they will increase at the same rate, resulting in a line at $y=x$. A model that captures some regularities in the data, however, will initially have a faster increasing true positive rate than a false positive rate because the high-scored utterances will tend to be grammatical ones. This results in a higher total area under the curve, a scalar metric that is often used as broad metric of the power of a binary classifier. This measue is closely related to precision and recall, but has the benefit of allowing interpolation between data points, resulting in a smoother curve [@davis06].


### Ordering a bag of words
As a second test, we use the task of ordering a bag of words as a proxy for production. A more direct test of production would be to generate utterances without any input, for example, by concatenating nodes in the graph based on transitional probabilities. However, this task has a number of disadvantages. First, it is difficult to evaluate the acceptability of generated utterances without querying human subjects. Second, utterance production in humans likely involves semantic as well as structural information, the first of which the present model attempts to capture. Thus, following previous work [@chang08; @mccauely14a], we treat a bag of words as an approximate representation of a thought a speaker wishes to convey, the ordering of the words being a more structure-specific task. To test the model, we take an unseen item from the corpus, convert it into a bag of words, and then compare the model's ordering to the original utterance.

A simple comparison strategy is to assign a score of 1 if the model's output perfectly matches the original, and 0 otherwise (the evaluation used in the aforementioned work). However, this metric selectively lowers the average score of longer utterances, which have $n!$ possible orderings. If the average score varies across utterance lenghts, utterances of different lengths will have varying discrimination power (in the extreme, no discrimination power if all models fail all utterances of a given length). Given this, we use the BLEU metric [@papineni02], which is more agnostic to utterance length. Specifically, we use the percentage of bigrams that are shared between the two utterances.

TODO equation for common_neighbor

## More results?
- CFGs of increasing complexity 
- contrived examples

## Introspection
- MDS of 10-50th most frequent words
- number of chunks over time

# Conclusion

