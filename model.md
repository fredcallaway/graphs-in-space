

# The model
This is a high level description of a proposed new model based on U-MILA. The representational strategy has changed significantly, but the core idea remains. Namely, the model builds linguistic knowledge by constructing increasingly large chunks of base tokens based on transitional probabilities between base tokens and previously identified chunks.

There are two main extensions to the new model:

1. It incorporates categorical knowledge in the form of Slots.
2. It combines learning and comprehension into one process model with cognitively meaningful states at each token.


## Representation
As in the original model, the model's knowledge takes the form of a Higraph. The Higraph is defined as a set of Nodes and several sets of directed edges, both weighted and unweighted. There are three types of Nodes: Tokens, Chunks and Slots.

<!-- $$FTP(A,B) = P(B_2|A_1) = \frac{P(B_2 \cap A_1)}{P(A_1)} $$
Forward Transitional Probability is the probability of another Node _following_ this Node. Here the subscripts indicate a Node being the initial or second element in a consecutive pair of Nodes in a Parse.

$$BTP(A,B) = P(B_1|A_2) = \frac{P(A_2 \cap B_1)}{P(A_2)} $$
Backward Transitional Probability is the probability of another Node _preceeding_ this Node.

Besides these two obligatory edges, a Node may have additional edges depending on its type. There are three types of Nodes: Tokens, Chunks and Slots. -->


### Token
_An atomic input unit._

This could be a single phoneme, a word, a syllable of bird song, or a location in space. It is given to the model as an atomic unit. Thus, the degree of abstraction of the Token object is the minimum degree of abstraction of the model.


### Chunk
_The basic structural unit._

A Chunk is sequence of Nodes. It is comparable to a linguistic constituent. The Chunk is wholly defined by its composition, represented as labeled "composition edges". In the present model, Chunks are required to have two such edges; thus, binary branching is enforced.

A Chunk is created when the model notices that two existing chunks occur adjacently more often than expected by chance. Both forward and backward transitional probabilities are taken into account.

\begin{framed}
\textbf{Should Chunks be flat or hierarchical?}
We could prevent Chunks from containing other Chunks, combining Chunks by concatenation rather than composition. This style does not rule out hierarchical output because a tree may be implicitly created in the process through which the model chunks an utterance. However, in both the original and proposed model, Chunks are allowed to contain Slots, which may themselves point to Chunks. It seems strange to allow a Chunk to include a Slot which may be filled by a Chunk, but not allow a Chunk to include a Chunk directly.

There is a tradeoff here between making the model more powerful (and perhaps resource intensive) in its procedural and declarative knowledege. Using hierarchical chunks may result in the model tracking separate transitional probabilities for one token sequence. However, if we do not make composition explicit, we require that the model be able to create the chunk from any two chunks that share a span (i.e. leaves) with the chunk. This is undesirable, especially if we want the merge operation to have semantics one day.

A further concern is that using hierarchical chunks makes it apper as though the models goal in parsing is to build a syntactic tree. This is not desirable, as the tree is meant to be a trace of the models parsing process, not a goal. In fact, the model will be prevented from accessing the elements of a Chunk.

The decision to use explicity hierarchical chunks comes mostly from the desire to align ourselves with existing parsing literature. By using composition, the Chunks become somewhat analagous to PCFG rules. This has the added benefit of allowing us to use previously developed parsing algorithms (a major plus from the programmer's perspective). In reality, this is not as large of a theoretical decision as it seems. Transferring from one strategy to another would not necessarily have a large effect on performance
\end{framed}


### Slot
_A probabilistic category of Nodes._

Categorical knowledge is represented with a Slot, which is comparable to a part of speech. A Slot has "filler edges", weighted edges to all Nodes that could fill the Slot. Each weight is a probability that the Slot be filled by the target Node. Thus, a Slot represents a distribution over Nodes.

In addition to representing intuitive categories such as Verb and Animate Object, Slots are used to represent transitional probabilities between Nodes. When a Node reaches a threshold weight, it gains two outgoing edges labeled "backward transitional probability" and "forward transitional probability". These edges point to newly created Slots which represent Nodes likely to occur before and after the Node. Nodes which have these two edges are called "tracked." 

\begin{framed}
\textbf{Set composition or set inclusion?}
We could prevent Slots from containing other Slots,  representing hierarchical categories through set inclusion rather set composition. This follows the original Higraph formulation more closely. However, this would prevent newly found information from percolating up into higher level categories. For example, we would want a set of food words to contain words that follow $eat$ and $cook$. Every time we see a word after either of these terms, we want the set of food words to be updated as well.

However, only using composition could result in unnecessary hierarchy. Thus, we may need to distinguish between cases where we care about information percolating up and those when we don't. In cases where we don't care, we can use an additive merge function. A simple, but possibly effective rule, is to use additive merge to add a single new item to a set, and composition merge to add two sets of cardinality > 1.

$$composition\_merge\big(\{A\}, \{B, C\}\big)  \rightarrow  \big\{\{A\}, \{B, C\}\big\}$$
$$additive\_merge\big(\\{A\\}, \{B, C\}\big)  \rightarrow  \big\{A, B, C\big\}$$
\end{framed}


\begin{framed}
\textbf{How should elements know what set they belong to?}
The parer must be able to assign categories to incoming Tokens and Chunks if it is to use categorical knowledge. This probably requires that Nodes have pointers to parent Slots. However, by representing transitional probabilities with Slots, we create a huge number of Slots, many of which will contain many nodes. <!-- FIXME: they haven't heard about representing transitional probabilities with slots yet --> We may want to avoid each Node having to point to every Slot that contains it. We probably don't want to have to update transitional probabilities for every Slot that contains each Node.

One possible solution is to only create element to Slot pointers for the most likely Slots. Thus, each Node will only know about the e.g. 10 Slots it's most likely to belong to. Another option is to point to all Slots, but update weights stochastically as opposed to updating weighted on the filler edge. That is, at each occurrence, the Node draws e.g. 10 Slots from its parent-slot distribution and only updates those 10 Slots.
\end{framed}

## Comprehension and Learning
In the original U-MILA, learning and comprehension were modeled separately.During the learning phase, the model would search backwards in memory for a newly completed chunk, and then update weights between each newly discovered chunk and the chunks ending immediately before the beginning of the newly discovered chunk. Comprehension was simplified into the task of assigning a probability to an utterance. This was done by summing across all traversals of the Higraph that have the utterance tokens as leaves.

### Problems with the original model
By allowing the model to look far back into memory to create chunks, the original U-MILA sacrifices considerable psychological plausibility. Experimental evidence indicates that humans cannot remember the order of even four sounds played in quick succession [@warren69].

The comprehension model of the original U-MILA (at least as described in the paper) does not constitute a process level analysis, nor does it explicitly relate to existing work in sentence comprehension. However, the basic idea of graph traversal can easily be reformulated as bottom up parsing, top down parsing being a relatively simple addition. Additionally, the co-occurrence based learning algorithm of U-MILA is straightforward to implement in such a system. (In fact, I guess that it would be easier to implement).

### Parsing
To the extent that the goal of U-MILA was to assign higher probability to grammatical sentences than ungrammatical sentences, the goal of the new model is to be able to chunk grammatical sentences, but not ungrammatical sentences. The model "chunks a sentence" when it is able to incrementally merge the sentence into larger chunks until it creates one chunk for the entire sentence. This is analagous to a parser creating a tree for an utterance.

<!-- Unlike in U-MILA, we require that the model build a structured representation of the sentence. This is desirable because it involves restricted access to memory and because itself to incremental semantic processing.

Unlike some other parsing models, we do not require that the model builds any particular type of structure. 
 -->

The models memory is a weighted set of stacks. Stacks contain at most STACK_SIZE elements. The model has three parsing operations. Note that the standard $reduce$ operation has been split into $merge$ and $categorize$. The ordering here reflects the order of operation, starting from $categorize$ each time the model progresses.

- $categorize$ replaces the top element of the stack with a Slot containing the element.
- $merge$ combines the top two elements of the stack into one Chunk. This operation can only occur if there exists a Chunk composed of the top two elements of the stack.
- $shift$ moves the next incoming Token onto the top of the stack. If doing so makes the stack more than STACK_SIZE elements long, the bottom element of the stack is removed.

There is indeterminacy in both the $merge$ and $categorize$ operations. The proposed model handles indeterminacy by using a set of stacks, rather than a single stack, where a stack is an intermediary parsing state. The StackSet is weighted, thus it can be viewed as a distribution over stacks. To prevent an exponential explosion of stacks, very unlikely parses are pruned from the representation. Jurafsky [-@jurafsky96] saw that contained parallel parsing can be modeled as *beam search* and we follow this strategy. We may additionally follow his dynamic programming strategy to minimize computation time, but this is not currently part of the cognitive model.

Note that the requirement that $merge$ takes two arguments means that trees will have binary branching. This is not so much a theoretical claim as much as a implementational requirement. If Chunks are created by composition, and only bigram probabilities are tracked, then there is no logical way to create a ternary branching Chunk (that I can come up with anyway).

However, I predict that the model will be able to represent something like ternary branching with two parses of one constituent. If the model has no preference between [[give me] that] and [give [me that]], we could say that the model is representing ternary branching using a distribution of binary branching parses. This also means that the model can represent varying degrees of ternary branching, in which one binary branching is preferred, but not completely domintant. Unfortunately, we cannot easily model the semantic aspect of binary branching because in either case the model must semantically combine two words first; a distribution of  combinations does not make as much sense with semantics as it does with syntax.

### Learning
The U-MILA learning algorithm can easily be adapted to fit into the beam search parsing algorithm. Because Chunks are held explicitly in memory, there is no need to search backward in memory for possible chunks. Rather, every time the stack is changed, the transitional probability links between the top two elements of each stack are increased. This update may be weighted by the probability of the stack (normalized by the sum of all stack probabilities), or it may occur stochastically based on stack probability. Notably, learning cannot be deferred to the end of an utterance because the model does retain access to elements that either fell off the stack or were incorporated into a larger chunk.

Another main proposal of the current model is in regard to how transitional probabilities are tracked. The original U-MILA model used edges directly from one Node to another. The proposed model represents transitional probabilities with a Slot. The outgoing temporal edges become slot filler edges, and an unweighted edges points from the Node to the slot, which can be labeled the "occurs after [node]" slot. Similarly, backward transitional probabilities are tracked with a "occurs before [node]" slot.

The main effect of this change is that temporal relationships and category membership relationships are represented with one tool: a distribution over Nodes. This allows categorical knowledge to depend entirely on temporal knowledge. (It also makes computational optimization much easier). The slots created to track transitional probabilities are primitive categories. Thus, category creation is broken into the easier task of combining existing categories. If two existing Slots have similar distributions, we create a new Slot pointing to the two SubSlots. For example, the "occurs after *this*" Slot and the "occurs after *that*" Slot will be very similar. We merge these into a new Slot that contains mostly adjectives and nouns.

\begin{framed}
\textbf{How do we constrain Slot creation?}
A major problem emerges from the strategy of representing transitional probabilities with Slots. Without preventative measures, Slots will recursively create themselves infinitely. If all Nodes received transitional Slots on creation, then the program would halt upon the first Slot creation. The new Slot would get a transitional Slot, which would get its own transitional Slot, and so on. Perhaps the simplest solution here is to differentiate between BaseSlots, those created for tracking transitional probabilities, and UberSlots, those created by combining BaseSlots.
\end{framed}

There is a lot more thinking to do on Slots...



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

# References