

# The model
This is a high level description of a proposed new model based on U-MILA. The representational strategy has changed significantly, but the core idea remains. Namely, the model builds linguistic knowledge by constructing increasinly large chunks of base tokens based on transitional probabilities between base tokens and previously identified chunks.

There are two main extensions to the new model:

1. It incorporates categorical knowledge in the form of Slots.
2. It combines learning and comprehension into one process model with cognitevely interpratable comprehension states at each token.


## Representation

As in the original model, the model's knowledge takes the form of a higraph. The Higraph is defined as a set of Nodes and several sets of directed edges, both weighted and unweighted. There are three types of Nodes: Tokens, Chunks and Slots.

<!-- $$FTP(A,B) = P(B_2|A_1) = \frac{P(B_2 \cap A_1)}{P(A_1)} $$
Forward Transitional Probability is the probability of another Node _following_ this Node. Here the subscripts indicate a Node being the initial or second element in a consecutive pair of Nodes in a Parse.

$$BTP(A,B) = P(B_1|A_2) = \frac{P(A_2 \cap B_1)}{P(A_2)} $$
Backward Transitional Probability is the probability of another Node _preceeding_ this Node.

Besides these two obligatory edges, a Node may have additional edges depending on its type. There are three types of Nodes: Tokens, Chunks and Slots. -->


### Token
_An atomic input unit._

This could be a single phoneme, a word, a syllabe of bird song, or a location in space. It is given to the model as an atomic unit. Thus, the degree of abstraction of the Token object is the minimum degree of abstraction of the model.


### Chunk
_The basic structural unit._

A Chunk is sequence of Nodes. It is comparable to a linguistic constituent. The Chunk is wholy defined by its composition, represented as labeled Composition Edges. In the present model, Chunks are required to have two such edges; thus, binary branching is enforced.

A Chunk is created when the model notices that two existing chunks occur adjacently more often than expected by chance. Both forward and backward transitional probabilities are taken into account.

<!-- **DECISION POINT:** We could allow Chunks to contain other Chunks, explicitly representing hierarchy Additionally, this style does not rule out hierarchical output because a tree is implicitly created in the process through which the model chunks an utterance. However, in both the original and proposed model, Chunks are allowed to contain Slots, which may themselves point to Chunks. It seems strange to allow a Chunk to include a Slot which may be filled by a Chunk, but not allow a Chunk to include a Chunk directly. -->


### Slot
_A probabalistic category of Nodes._

Categorical knowledge is represented with a Slot, which is comparable to a part of speech. A Slot has "Filler Edges", weighted edges to all Nodes that could fill the Slot. Each weight is a probability that the Slot be filled by the target Node. Thus, a Slot represents a distribution over Nodes.

In addition to representing intuitive categories such as Verb and Animate Object, Slots are used to represent transitional probabilities between Nodes. When a Node reaches a threshold weight, it gains two outgoing edges labeled "backward" and "forward." These edges point to newly created Slots which represent Nodes likely to occur before and after the Node. Nodes which have these two edges are called "tracked." 

<!-- The Slot is comparable to a set in a standard higraph. As in a standard higraph, Slot hierarchy is represented as set inclusion rather than set membership. However, because set membership is probabalistically weighted, the subset relationship is not clearly defined. -->


\begin{framed}
\textbf{Set composition or set inclusion?}

We could prevent Slots from containing other Slots,  representing hierarchical categories through set inclusion rather set composition. This follows the original Higraph formulation more closely. However, this would prevent newly found information from percolating up into higher level categories. For example, we would want a set of food words to contain words that follow $eat$ and $cook$. Every time we see a word after either of these terms, we want the set of food words to be updated as well.

Only using composition could result in unnecessary hierarchy. Thus, we may need to distinguish between cases where we care about information percolating up and those when we don't. In cases where we don't care, we can use an additive merge function. A simple, but possibly effective rule, is to use additive merge to add a single new item to a set, and composition merge to add two sets of cardinality > 1.

$$composition\_merge\big(\{A\}, \{B, C\}\big)  \rightarrow  \big\{\{A\}, \{B, C\}\big\}$$
$$additive\_merge\big(\\{A\\}, \{B, C\}\big)  \rightarrow  \big\{A, B, C\big\}$$
\end{framed}


\begin{framed}
\textbf{How should elements know what set they belong to?}

The parer must be able to assign categories to incoming Tokens and Chunks if it is to use categorical knowledge. This probably requires that Nodes have pointers to containing Slots. However, by representing transitional probabilities with Slots, we create a huge number of Slots, many of which will contain many nodes. We may want to avoid each Node having to point to every Slot that contains it. We probably don't want to have to update transitional probabilities for every Slot that contains each Node.

One possible solution is to only create element to Slot pointers for the most likely Slots. Thus, each Node will only know the e.g. 10 Slots it's most likely to belong to. Another option is to point to all Slots, but update weights stochastically as opposed to updating weighted on the filler edge. That is, at each occurrence, the Node draws e.g. 10 Slots from its parent-slot distribution and only updates those 10 Slots.
\end{framed}

<!-- ## Learning algorithm -->


<!-- 
### Parse
_A sequence of Nodes and a probability of the sequence occuring._

This corresponds to a parse tree in a typical parsing framework. The probability of a Parse is currently modeled as the probability of the first element times the product of forward transitional probabilities between all other elements in the list. This could be extended to take into account backward transitional probabilities. It could also be extended to take into account the probability of each Node in the parse

### ParseSet
_A set of parses with the same surface string._

This objs necessary. Thus, parsing with a ParseSet is very similar to Beam Search.

This is currently implemented as a set, hoect represents the short term memory of the model. The model's representation of a series of tokens (i.e. an utterance) is fully contained in a ParseSet for that utterance. This maps onto a common psycholinguistic theory that the human parser tracks multiple possible parses of a sentence in parallel. The ParseSet can contain no more than *ParseSetSize* parses at any point. Thus, improbable parses are pruned awever this ignores the overlap among parses. Thus, the model's working memory may be better represented as a decision tree where the Parse associated with each node is constructed by following the tree from the root to the given node. This would allow a more realistic memory constraint: the number of nodes in the decision tree.


## Learning and parsing algorithm
In the present model, learning and parsing occur simultaneously. Learning consists of updating edges in the graph and creating new Nodes. Below, we list some high level characteristics of the model, and then provied pseudocode.

- The model performs right corner parsing at each new token
- The model tracks probability of each parse at each step using forward transitional probabilities of the top level Nodes in the parse (the Markov assumption). 
- The model tracks the most probable _ParseSetSize_ parses of the in-memory sequence.
- Edge weights between Nodes are updated, weighted by the probability of the parse.

```
read token
for each Parse in ParseSet
    # find all new Parses given this Token
    append Token to Parse
    multiply probability of Parse by FTP of two most recent Nodes
    while the last two Nodes in Parse can form a Chunk
        # make a new Pars with this Chunk
        remove the most recent two Nodes from Parse
        divide probability by FTP of each Node with its preceding Node
        append Chunk to Parse
        multiply probability by FTP of the two most recent Nodes
        # the last two Nodes have now changed, so we check again
remove all but the top ParseSetSize 
for each Parse in ParseSet
    increase collocation count of the last two Node in Parse 
      weighted by P(Parse|utterance)
    if the collocation count meets some threshold (e.g. Barlow)
        create a new Chunk from last two Chunks
        add new Chunk to Higraph
```

Calculating FTP
  

Parsing
    for each token in utterance  
        for each parse  # O(n) where n is the number of parses in the ParseSet
            1. check if most recent two elements can form a chunk  # one string concatenation + one hash lookup = very fast
            2. multiply by FTP(B, C)  # two hash lookups + one multiplication = pretty fast
            if step 1 returns true  # O(p) where p is the probability that two consecutive chunks will form a chunk
                3. copy the parse  # O(k) where k is the number of elements in parse
                4. divide by FTP(A, B)  # two hash lookups + one division = pretty fast
                5. multiply by FTP(A, [B C])  # one hash lookup + one multiplication = pretty fast
                6. go back to 1  # O(h) where h is max hierarchical depth

Updating
for each parse  # O(n)
    for each element in parse  # O(k)  

## Thoughts

### Weighting collocations by probability
Does it make sense to weight collocation updates by the probability of the parse? This results in a "rich get richer" situation where frequent chunks will have their weight increased more quickly. This is especially problematic in how it favors old vs. newly discovered chunks. At the same time, it seems unideal for a chunk to become heavily weighted because it occurs next to the boundary of several chunks in improbable parses.

I've listed some ideas to combat this problem below, but none are quite satisfactory to me.

- bumping probability of new chunks (a newly discovered chunk has increased activation---sort of makes sense)
- an initial learning phase which doesn't add weights to collocations, or perhaps considers weights less strongly.
- a global preference for more chunks in a parse (motivated perhaps by the idea that a more hierarchical parse better represents the semantic structure of the sequence)

### Count collocations for utterances only?
If we give the model an innate notion of utterance, we have an alternative weight update strategy. Instead of updating at each step, we could update once at the end of an utterance. This would be good for time complexity because it lowers the total number of updates that need to be made. Additionally, it might not make sense to update mid constituent, when probability of the parse will not be as meaningful. Finally, it removes the current asymmetry where the initial words affect the probability (and hence the weight update) of the final chunks but not vice versa.

### Trigrams?
Currently the parsing model is a first order Markov process over chunks. Could we extend it to a second order Markov process? We would need to track a three dimensional collocation matrix. It would also complicate the simple notions of FTP and BTP. However, The model is currently restricted to binary branching, which will result in very hierarchical representations. This seems undesirable for constructions like "give ___ to ___" which really seem to involve three or four elements. In the current model are trigram chunks only created via top down phonoloop chunking? 

In a similar vein, should unigram probabilities be considered? That is, if a chunk __[A B]__ occurs frequently but never yet after an infrequent token __D__, should the frequency of __[A B]__ matter at all?

### BTP in parse probability
Having different rules for parse probability (only FTP) and chunk creation (FTP and BTP) feels less than ideal. Perhaps we could track parse probability by a combination of FTP and BTP as well? I don't see any computational reason this couldn't be done. Doing so could simplify the model by further unifying the parsing and chunk creation processes. We could also get some cool results comparing optimal weights for left branching and right branching languages.

### Emulating a phonoloop with activation
Currently the model does not implement anything comparable to the phonoloop in U-MILA. This is partly because I doubt the psychological plausibility of holding 50-300 tokens in memory at a time, let alone the task of searching such a long sequence for repeated elements. Furthermore, the division into two completely separate chunking strategies complicates the model, perhaps unnecessarily.

I wonder if we could achieve a similar function by using the standard bottom up chunking model in conjunction with residual activation effects. Upon hearing two words next to each other, their link becomes activated. If this link is activated again in a short time window, then it will exceed threshold and the two words will be chunked.

Along with the previous idea of using BTP in parse probability, we may be able to unify (i) probability assignment, (ii) bottom up chunking, and (iii) top down chunking into one general model.

### Chunk categories
My initial idea to expand U-MILA was to add chunk categories, (like parts of speech in language). I imagined them as slots in U-MILA, but not needing to be tied to a chunk. Currently, categories are modeled implicitly through their similarity patterns. However the linguist in me wants these to be explicit constructions used by the model.

By explicitly modeling categories, and the transitional probabilities between them, I predict that the model will be able to learn more complex structures than otherwise possible. This is because categories will occur more frequently than basic chunks, thus there will be more probabilistic information for their transitional probabilities.

One mian concern here is that this this could potentially result in a huge number of categories to track in the Higraph. We will also need a mechanism to discourage a category from becoming too broad because, as specified here, a category that included all words would occur constantly, increasing its weight to infinity in a feedback loop.

### Predictive parsing
The final, and perhaps most ambitious, extension I propose is to incorporate predictive parsing. There is a consensus in psycholinguistics that humans form some representation of upcoming words, thus it would be good for the model to reflect this. Additionally, there are computational costs associated with right corner chunking such as following parses that are very improbable given the larger structure. 

A guess as to how to implement this: if the weight from the current chunk to another chunk is very high, then the model predictively creates that chunk. The model attempts to fit upcoming tokens into that chunk until the chunk is completed or the parse becomes too improbable. This will mostly only work with categories because the model will almost never be certain enough to predict an exact token.

 -->