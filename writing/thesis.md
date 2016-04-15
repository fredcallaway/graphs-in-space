---
title: "Representing graphs with sparse vectors."
author: Fred Callaway
date: \today
---

# Introduction
Structure learning is a fundamental task for cognitive systems, and identifying mechanisms for structure learning is a fundamental task for cognitive scientists.

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

### Graphical models and traditional syntactic theories

Network theories of language often suggest that language is most fundamentally described by relationships between individual words [@hudson03], while traditional linguistic theory emphasizes composition and relationships between constituents [@chomsky65; @chomsky95; @stabler96]. Although these views are frequently seen as opposing, they may serve complimentary roles. Perhaps direct word-word relationships characterize early linguistic knowledge that is only later developed into more complex syntactic structures [@bannard09]. It is also possible that the representations of formal linguistics should be viewed as theoretical abstractions that roughly characterizes a fuzzier, network-based language system [@lamb99]. @marr82 calls Chomsky's (1965) theory of transformational grammar a "true computational theory", suggesting that Chomsky's performance-competence distinction reflects Marr's algorithmic-computational distinction. Under this interpretation, network and phrase structure theories may be posed at different levels of analysis; if the theories make roughly similar predictions, they may not conflicting.


# BindGraph (sorry)
We describe an augmented graphical datastructure, intended to represent domains that are defined by complex inter-relationships and compositional structure. Language involves more than the relationships between words and constituents [citation]; however, the structural aspects of language may be modeled well with such a datastructure. The fundamental principle of a BindGraph are (1) elements are defined by their binary relationships with other elements, and (2) elements can be meaningfully combined to form new elements. Both of these principles have a long history in linguistics. @firth57 observed that "you shall know a word by the company it keeps" (p. 11), and perhaps additionally by "how it keeps it" [@jones07 p. 2]. In theoretical syntax, dependency grammars [citation] and combinatory categorial grammar [@steedman00] are both based on the relationships between individual elements (although moderated by categories).

The notion of composition has an even richer history in linguistics. Nearly all modern syntactic theories take some sort of binding operation as the fundamental operation. The operation may be called "Merge" [@chomsky95], "(function) application" [@steedman00], or "Unification" [@hagoort04]; however, in each of these frameworks (with widely varying theoretical motivations), the principle of combining two elements to form one element remains. Despite this similarity, these three frameworks use widely differing representational machinery, and as a result, it is sometimes difficult to specify the differences and similarities between theories. Graphs may provide a general representational framework with which different linguist theories could be compaered.

A BindGraph is based on a weighted, labeled, directed, multigraph. That is, a node may have multiple edges pointing to another node, representing different types of connections. Additionally, nodes in a BindGraph may be composed of other nodes, inspired by Higraphs [@harel88]. For example, the constituent `[the dog]` could be a single node, composed of the nodes `the` and `dog`. This node could have two edges to the node `scared` labeled _argument_ and _subject_.

A BindGraph is a tuple $G = TODO$ where

- $V$ is a finite set of vertices.
- $\Sigma_E$ is a finite set of edge labels.
- $E$ is finite set of edges, each of which is a tuple $(x, y, \ell)$ where $x$ and $y$ are vertices, and $\ell$ is label $\in \Sigma_E$
- $fweight$ is a function where $fweight(x, y, \epsilon)$ is the edge weight of type $\epsilon$ from $x$ to $y$
- $fbump$ is a function where $fbump(X, Y, \epsilon)$ increases the edge weight of type $\epsilon$ from $X$ to $Y$
- $fbind$ is a function where, for a list of nodes $\mathbf{x}$, $fbind(\mathbf{x})$ is a node $y$ where $y$ represents the ordered composition of the nodes in $\mathbf{x}$.


[TODO Not sure how to formally describe a mutable datastructure. Note that the node actually contains all its edges, which is what makes bind work. A node can have outgoing edges without being "in" the graph.]

Like a standard graph, BindGraph is an abstract data type, meaning it can be implemented in a number of ways. In the case of a BindGraph, however, the implementation can have a profound impact on the behavior. In particular, the bind function plays a major role in the intelligence of the graph. This function can be specified by the modeler or it can be learned.



# Graphs in space
We describe an implementation of a graph using vectors in a large, fixed-dimension space. The motivation for such an implementation is two-fold. First, recent years have seen promising work in composition operations for vectors, much of it with a focus on semantic composition in language [see @mitchell10 for a review]. More generally, fixed-dimension vectors (e.g. feature vectors) are the basis of many machine learning algorithms; representing a graph with such vectors allows a modeler to draw on this work, for example when constructing an algorithm to learn a binding function. Second, any model that employs distributed representations bears a closer resemblance to brains, which are fundamentally distributed computers. Implementing a symbolic graph with distributed vectors may provide a bridge between the symbolic representations preferred by linguists and the distributed representations proposed by the PDP and Connectionist frameworks [see @smolensky06 for an alternative approach].

TODO expand on each of the above points?

To construct a spatial representation of a graph, we begin with the traditional adjacency matrix. Noting similarities between this matrix and the co-occurrence matrices employed by distributional semantic models. Inspired by recent work in this field, we constrcut an approximate representation of an adjacency matrix using _random indexing_, an incremental and efficient method for approximating a large, sparse matrix [see @sahlgren05 for an accessible review]. The resulting data structure closely mimics the behavior of an adjacency matrix representation, but additionally affords transformation, similarity, and composition operations defined in fixed-dimension spaces.


## Distributional semantic models
The core idea underlying distributional semantic models such as HAL [@lund96], LSA [@landauer97], and Topic Models [@griffiths07], is that a words meaning can be approximated by the contexts in which it is used. The data come in the form of a very large and sparse matrix, with one row for each word and one column for each document (or word, depending on how context is defined). The row can be interpreted as a feature vector specifying the word's meaning, and this vector resides in a very high dimensional word/document space. This space is, in fact, so large that the raw vectors are too large to effectively use. To address this, distributional models employ some form of dimensionality reduction such as singular value decomposition. However, this operation is very costly and it must be rerun from scratch if one wishes to add more data. This is not so problematic in an engineering context, when a model can be trained once and used many times. However, as a cognitive model, batch processes such as these leave much to be desired.

@kanerva00 demonstrate that random indexing can be used as an efficient, online dimensionality reduction tool for distributional models. Rather than constructing the full word by document matrix and then applying dimensionality reduction, this technique avoids building this matrix to begin with. The number of columns is fixed ahead of time to be some constant $D << N$ (where $N$ is the number of documents). Each document is then assigned an _index vector_ which is a sparse ternary vectors (i.e. containing many 0's and a few 1s and -1s). The rows of the matrix, called _context vectors_ are produced by adding a document's index vector to the context vector of every word occurring in the document. That is, a word's context vector is the sum of the id vectors for every document it occurs in. This technique has been found to produce similar results to LSA at a fraction of the computational cost [@karlgren01].

A similar approach is taken by @jones07, who describe a method for incorporating order information into distributional semantic models using holographic reduced representations [@plate95]. This work makes the important contribution of including multiple types of information in one vector. The BEAGLE model encodes N-grams of various sizes surrounding a word using circular convolution with permutation [to preserve order, as suggested by @plate95].

## Random indexing for graphs
A standard representation of a graph is an adjacency matrix. This matrix is $M_{N \times N}$, where each row represents the outgoing edges of one node. Applying this interpretation to the co-occurrence matrix used in a word-word distributional semantic models, we have a graph with words as nodes and co-occurrence counts as edge weights. If a co-occurrence matrix can be interpreted as a graph, and a co-occurrence matrix can be approximated with random indexing, perhaps a graph can also be approximated with random indexing. Indeed, viewing a co-occurrence matrix as a special case of a graph, this is simply a generalization of the random indexing technique for co-occurrence graphs to any other kind of graph.

We can construct such a generalization by directly mapping elements of Kanerva's algorithm to elements of a graph. Context vectors, which represent the meaning of a word, become _row vectors_, which represent all outgoing edges of a node. Just as each word/document receives an index vector, each node receives an index vector. To increase the weight of the link from $x$ to $y$, we add $id_y$ to $row_x$. Similarity between nodes is defined as cosine similarity. Nodes that have similar outgoing edges, or _edge profiles_, will be similar because their row vectors will contain many of the same id vectors. Importantly, because random vectors in a high dimensional space tend to be nearly orthogonal, row vectors for nodes that share no outgoing edges will have similarity close to 0. Additionally, we can use the cosine operation between a $row_x$ and $id_y$ to recover the edge weight of $x$ to $y$.

An interesting attribute of this representation is that edge weights behave somewhat like probabilities. That is, increasing the weight from $x$ to $y$ will slightly decrease the weight from $x$ to $z$. Visually, $y$ is pulling $x$ towards it, and thus away from $z$. However, unlike probabilities, there is no hard bound on the sum of all edge weights for a node. Total edge weights increase as number of outgoing edges increase, but at a deccelerating rate, as shown in figure #.

![Total edge weight as a function of number of edges. Vectors are length 500.](/path/to/img.jpg)

## Design choices
We are presented with some design choices when implementing a MergeGraph with vectors. A key choice lies in how edge labels are implemented. One option is to encode all edges in one vector, using permutations to differentiate edges with different labels. @basile11 use this technique to encode syntactic dependency information in a spatial semantic model. Specifically, each dependency relation (e.g. _subject_) receives a unique permutation vector, which is used to permute a word's index vector before adding it to the context vector. This method has the advantage of keeping the dimensionality constant.

However, the additive combination of edges of different type has some side effects. First, they will compete with each other--if one edge type gets used more frequently, it will overpower the other, and may even destroy the other with noise. Secondly, any similarity computation will use additive feature combination. This is problematic in settings where the _intersection_ of features is critical. For example, while measuring the similarity of "the" and "me", an edge type that points to commonly preceding elements will have similar profiles for both words. This could result in "the" and "me" being classified as the same part of speech. If edges with different labels were stored separately, on the other hand, we could measure the similarity for each separately and take the product of similarities, which would be low if an edge point to commonly following elements was included. [TODO this is messy. There's a great Levy citation here I can't find.]

- convolution vs permutation and addition


# Nümila
In order to test the sparse vector graph representation, we created a simple language acquisition model, Nümila, based on previous graphical models [@solan05; @kolodny15]. Like these previous models, the present model reduces the problem of language acquision to the much simpler problem of producing grammatical utterances based on statistical patterns in speech. In reality, language acquisition is heavily dependent on the semantic and social aspects of language [@tomasello03; @goldstein10], aspects which the present model does not capture. However it is generally agreed that linguistic pattern recognition plays at least some role in language acquisition; thus, the present model can be seen as a baseline that could be improved upon by enriching the input to the model with environmental cues. TODO We discuss ways that these cues could be incorporated into a graphical model.

## Graphical model

### Edges
The model has two edge-types representing forward and backward transitional probabilities. Although forward transitional probability (FTP) is the standard in N-gram models, some evidence suggests that infants are more sensitive to BTP [@pelucchi09], and previous language acquisition models have been more successful when employing them [@mccauley11]. To examine the relative contribution of each type of TP, we make their relative weight an adjustable parameter. Although ADIOS and U-MILA have only one type of temporal edge (simply a count of coocurrences), their learning algorithms compute something very similar to FTP and BTP [TODO proof?]. By using two edge types, we build this computation into the representational machinery.


### Merge
When two nodes (initially words) are determined to cooccur at an unexpectedly high frequency (see below), the graph's bind function is applied to create a new node. As discussed above [TODO], the bind function, $fbind$, is a parameter of the graph, and must be a function from a list of nodes to a single node. As a simplifying assumption, we follow U-MILA by only considering binary binds; thus the function is of type $N \times N \rightarrow N$. Importantly, we do not make the theoretical claim that linguistic composition must be binary [as others have @everaert15]. We implement two such bind functions, one hierarchical and the other flat. Given arguments `[A B]` and `[C D]`, hierarchical bind returns `[[A B] [C D]]`, whereas flat bind returns `[A B C D]`.

In the simplest case, the bind function determines only the identity of the new node. However, this quality alone has important ramifications. Hierarchical bind is a bijective function; that is, there is a one-to-one mapping from inputs to outputs. Conversely, flat bind is not bijective because multiple inputs can produce the same output. For example, if `[A B C]` and `D` occur together frequently, a new node `[A B C D]` will be created and added to the graph. Later on, if `[A B]` and `[C D]` are bound, we will get the existing node, `[A B C D]` with all its learned edge weights. In more practical terms, a model using a flat bind rule will treat every instance of a given string e.g. "Psychology department chair" as the same entity. Although there are clear semantic reasons why a flat bind function would be undesirable, it is not clear to what extent hierarchical information will be useful for the purely grammatical tasks we test our models with TODO.

The full power of the bind function, however, comes from its ability to construct initial edge weights for the new node. This allows truly compositional structure, where the behavior of larger units is predictable based on its constitutents. This is critical for syntactic processing: Even if you had never heard the phrase "honest politician", you could still predict its syntactic behavior. In a graph, the syntactic behavior of a node is represented in its edge weights. Thus, predicting syntactic behavior comes down to constructing an initial edge profile for the newly created node based on the edge profiles of its element nodes. This function can be specified by the modeler or learned. [TODO specified bind function]


## Learning

The model constructs a Higraph given an initially blank slate by processing each utterance, one by one. Thus, the model has more limited memory resources than both ADIOS and U-MILA. The graph is constructed with three operations: (1) adding newly discovered base tokens to the graph, (2) increasing weights between nodes in the graph, and (3) creating new nodes by binding existing nodes. We implement two processing algorithms that employ these basic operations to learn from an utterance. The first is meant to replicate U-MILA's bottom up chunking mechanism, learning transitional probabilities between all possible nodes in the utterance. The second is inspired by the Now-or-Never bottleneck [@christiansen15], incorporating an even more severe memory constraint, and building up a structural interpretation of the utterance word by word.

<!-- 
The core of the learning algorithm lies in (1) updating the weights between adjacent pairs nodes, and (2) considering the binding of adjacent pairs of nodes. The first consists of increasing  $E_F(X, Y)$ and $E_B(Y, X)$.
 -->

### FullParse

The FullParse algorithm is similar to U-MILA's bottom up learning algorithm in that it finds all pairs of adjacent nodes (potentially overlapping) in the utterance and then applies a learning process to the pair. The algorithms differ in the edge weights that are updated, and the criteria for creating a chunk. For every pair of nodes $(X, Y)$ such that $X$ directly precedes $Y$ in the utterance:

#. Increase weights
    #. Forward transition probability $E_F(X, Y)$
    #. Backward transition probability $E_B(Y, X)$
#. Attempt to create a chunk
    #. Check that binding the pair would not result in a node already in the graph
    #. Check that the _chunkiness_ of the pair exceeds a fixed threshold, where _chunkiness_ is the geometric mean of the forward and backward transition probabilities between the nodes: $\sqrt{E_F(X, Y) \cdot E_B(Y, X)}$
    #. Create a new node by binding $X$ and $Y$
    #. Add this node to the graph

Unlike U-MILA's algorithm, this algorithm requires the full utterance to be in memory before any processing is done. This is because the specific algorithm isn't meant to be cognitively realistic [TODO should this be discussed, perhaps footnoted?]

### GreedyParse

The GreedyParse algorithm follows the same basic principles as FullParse in that it is based on updating transitional probability edges and binding nodes. However, unlike FullParse, GreedyParse incorporates severe memory limitations and online processing restraints in line with the Now-or-Never bottleneck [@christiansen15]. In contrast to FullParse, which finds all possible structures for an utterance given the current graph, GreedyParse finds a single structure by making a sequence of locally optimal decisions, hence "Greedy". Upon receiving each word it can create at most one chunk and the nodes used in this chunk can not be used later in a different chunk. Thus, the algorithm may not assign the optimal structure to the utterance.

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
    create chunk by binding the two nodes
    if chunk is not in graph:
        add chunk to graph
        update weights between new chunk and adjacent nodes
    fi
else:
    remove the oldest node from memory
fi
```

### Generalization
The model is essentially a bigram models with variable length elements. Thus, like any N-gram model, it will suffer from the problem of data sparsity. Incorporating variable-length units increases this problem dramatically because the number of tracked elements is no longer limited to the vocabulary size. A generalization strategy is essential to counteract this sparsity problem. In a classical linguistic view, generalization is accomplished with categories such as VERB and NOUN. The reality of parts of speech is a basic assumption in linguistics, and acquisition models often use parts of speech as input [@bod09; TODO]. The problem of learning these categories, on the other hand, has received less attention in linguistics. Algorithms for part of speech induction based on distributional statistics have had some success [@schutze95; @clark00]. However these models come from natural language processing, employing computationally intensive employing batch processing algorithms.

Some work has sought to connect POS induction models with syntax induction models, learning language structure in stages [@klein05]. However, as a cognitive model, this approach assumes that word categories are learned and finalized before any syntax is learned. While category learning may begin earlier, there are good reasons for the learning of categories and syntax to occur simultaneously. Seeing as a large motivation for categories is the role they play in syntactic pattern recognition, it would be desirable for the syntactic learning process to affect categoy learning. While linear context may provide a decent cue to syntactic category, the role a word plays in higher level synstactic patterns may hold more information. For this reason, we pursue a single stage approach in which word-categories and phrasal patterns are learned in tandem.

Another common assumption that deserves examination is the one that syntactic categories are explicit and distinct. Having suffered through explicit instruction on the rules of grammar, most students are able to classify most words into the basic parts of speech. However, it is not clear to what extent these categories are the product of basic language acquisition as opposed to academic analysis. Furthermore, even for the student who can correctly label words, it is unclear to what extent this knowledge is called upon in every day language use.

- exemplar/prototype?

### Compositionality
Presumably, experience with pairs of similar words informs your expectations for how the syntactic role of this novel phrase. The standard linguistic explanation involves rules and categories: "honest" is an `Adj`; "politician" is a `N`; and thus by the rule `N' -> Adj N`, "honest politican" is an `N'`. The extensive regularity of language implies that such theories are at the very least a useful computational-level theory [as @marr82 suggests]. However, it is an open question whether adult speakers truly represent rules and discrete categories. It's possible that rule-like behavior emerges from a system that explicitly represents only relationships between individual items [citation].

- rules and hard categories not characteristic of brains
- computational vs algorithmic
- how to learn?

perhaps based on your experience with other adjective-noun pairs.  the new nodes edges would be similar to other nodes that were composed of similar elements.

- talk a lot about baroni


# Testing the model
To test the model, we use naturalistic child directed speech. We use corpora prepared by @phillips14, which include syllabified and word-tokenized versions of CDS corpora in 7 languages. All models are trained on the first 4000 utterances of the respective corpus. We test several instantiations of Nümila using different BindGraph implementations, parsing algorithms, and bind functions.

## Experiment 1: Grammaticality judgement
As a first test, we use the common task of discriminating grammatical from ungrammatical utterances. This task is appealing because it is theory agnostic (unlike evaluating tree structures) and it does not require that the model produce normalized probabilities (unlike perplexity).

### Generating an acceptability score
Statistical language modeling is sometimes equated with determining the probability of word sequences [c.f. @goodman01], something that Nümila cannot easily do given that outgoing edges for one node (labeled e.g. FTP) are not required to sum to 1, and are thus not probabilities. However, many tasks that employ utterance probability can be performed just as well with scores that are only proportional to a true probability.

In a generative language model, the probability of an utterance is the sum of the probabilities of all possible ways to produce the utterance (e.g. all tree structures). The probability of each structure is the product of the probabilities of every rule that is applied in creating the utterance. With a PCFG, the rules are all of the form $NT \rightarrow \alpha$, where $NT$ is a nonterminal symbol (representing one branch of the tree, a constituent) and $\alpha$ is a sequence of symbols, either terminal or nonterminal. With an N-gram model, on the other hand, the rules are all of the form $\alpha \rightarrow \alpha \cdot w$, where $\alpha$ is the $n-1$ most recent words and $w$ is the next word.

Because Nümila incorporates structural elements (chunks) and transitional probabilities, it uses both types of rules. For chunks, the PCFG rule is applied; however, because each node has exactly one compositional structure, the rule probability is always 1. When an utterance has a series of nodes that cannot be combined, the bigram rule is applied: For each adjacent pair of nodes, $(X, Y)$, we apply the rule $X \rightarrow X \cdot Y$ with probability proportional to $E_F(X, Y)$. The result is simply the product of FTPS between each pair of nodes spanning the utterance. [TODO similar to something, but what?!] Finally, to avoid penalizing longer utterances, we take the result to the $n-1$ root where $n$ is the length of the utterance.

![An example utterance with score = $\sqrt[5]{E_F(A, [B C]) \cdot E_F([B C], [D [E F]])}$](figs/tree.png)

Given a function that assigns scores to a single parse of an utterance, it is straightforward to create a function that assigns scores to the utterance itself. With a PCFG (where the scores are probabilities) the probability of the utterance is the sum of the probabilities of each parse. This is a result of the assumption that the utterance is generated by the PCFG model, along with Kolmogorov's third axiom that the probability of the union of independent events is the sum of the probabilities of each event. Although Nümila's scores are not true probabilities, we apply the same rule. That is, the score of an utterance is the sum of the scores for all parses of that utterance. [TODO does this maintain proportional probability?]

### Preparation of stimuli and analysis of performance
To construct a test corpus, we first take # unseen utterances from the corpus, which are labeled "grammatical". For each utterance, we create a set of altered utterances, each with one adjacent pair of tokens swapped. For example, given "the big dog", we create "big the dog" and "the dog big". These altered utterances are added to the test corpus with the label "ungrammatical". The models task is to separate grammatical from ungrammatical. Often, this task is modeled by setting a threshold, all utterances being predicted to be grammatical. However, it is unclear how to set such a threshold without either specifying it arbitrarily or giving the model access to the test labels. Thus, we employ a metric from signal detection theory, the Recevier Operator Characteristic.

The ROC curve plots true positive rate against false positive rate. As the acceptability threshold is lowered, both values will increase. With random scores, they will increase at the same rate, resulting in a line at $y=x$. A model that captures some regularities in the data, however, will initially have a faster increasing true positive rate than a false positive rate because the high-scored utterances will tend to be grammatical ones. This results in a higher total area under the curve, a scalar metric that is often used as broad metric of the power of a binary classifier. This measue is closely related to precision and recall, but has the benefit of allowing interpolation between data points, resulting in a smoother curve [@davis06].

### Results


![ROC curve on the English word corpus.](figs/roc-curve.pdf)
![Area under ROC curve for different input types, collapsed across laguages.](figs/roc-type.pdf)

## Experiment 2: Production
As a second test, we use the task of ordering a bag of words--a proxy for production. A more direct test of production would be to generate utterances without any input, for example, by concatenating nodes in the graph based on transitional probabilities. However, this task has two disadvantages. First, it is difficult to evaluate the acceptability of generated utterances without querying human subjects. Second, utterance production in humans likely involves semantic as well as structural information, the first of which the present model does not attempt to capture. To avoid these problems, we follow previous work [@chang08; @mccauley14a] by using a word-ordering task to isolate structural knowledge. A bag of words is taken as an approximate representation of the thought a speaker wishes to convey; speaking then becomes simply the task of saying the words in the right order.

### Ordering a bag of words
We treat ordering a bag of words as an optimization problem, using the acceptability score described above as a utility function. The optimal but inefficient strategy is to enumerate all possible orderings of the words and choose the one with the highest acceptability score. However, with $n!$ possible orderings, this becomes intractable for longer utterances. As with learning, we propose a greedy algorithm to approximate the optimal solution. Typically such an algorithm starts from the beginning of the utterance and iteratively appends the best word or chunk to the end, producing the utterance in the same order it was spoken [e.g. @mccauley14a; @kolodny15]. This parallel has some theoretical appeal, however, it not clear that utterances are planned in the same order that they are spoken. Indeed, research in predictive sentence processing indicates that listeners actively predict upcoming clauses [citation]--it is thus reasonable to think that speakers may plan ahead beyond the next few words. Lacking strong theoretical motivation for purely incremental sentence planning, we explore a more flexible approach.

The algorithm begins by greedily constructing chunks using the input nodes. When no more chunks can be made, the chunks are combined to form an utterance. This is done by iteratively adding a node to either the beginning or the end of the utterance, whichever maximizes chunkiness between the new adjacent pair.

```
# Create chunks.
while a chunk can be made:
    select pair of nodes with highest chunkiness
    if chunkiness < threshold:
        break while
    replace the pair with the chunk constructed by binding the pair

# Create utterance.
utterance = []
add chunk with highest chunkiness to utterance
while there are nodes left:
    select the node that has the highest chunkiness with either
      the first or last element of the utterance
    add the node to the beginning or end of the utterance
```

### Preparation of stimuli and analysis of performance
To test the model on this task, we take an unseen item from the corpus, convert it into a bag of words, and then compare the model's ordering to the original utterance. A simple comparison strategy is to assign a score of 1 if the model's output perfectly matches the original, and 0 otherwise [as in @mccauley14a]. However, this metric selectively lowers the average score of longer utterances, which have $n!$ possible orderings. If the average score varies across utterance lenghts, utterances of different lengths will have varying discrimination power (in the extreme, no discrimination power if all models fail all utterances of a given length). Given this, we use the BLEU metric [@papineni02], which is more agnostic to utterance length. Specifically, we use the percentage of bigrams that are shared between the two utterances.

### Results
The current results are placeholders.

![Production results.](figs/production.pdf)

## Experiment 3: Generalization



<!-- 
Recall that when an utterance is parsed using the GreedyParse algorithm, only one possible parse is found. Thus, the score of the utterances is simply the score of that single parse. As a consequence, the GreedyParse algorithm prefferentially treats utterances that have a narrower, or lower-entropy, distribution of scores for all possible parses. Conversely, an utterance that has many possible parses, all with somewhat high scores, will be at a disadvantage.
 -->





## More results?
- CFGs of increasing complexity 
- contrived examples

## Introspection
- MDS of 10-50th most frequent words
- number of chunks over time

# Conclusion
This was fun.

# References
