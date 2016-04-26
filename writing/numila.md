
To demonstrate how a BindGraph can be used as a component of another model, we created a simple language acquisition model, based on previous graphical models [@solan05; @kolodny15] and chunking models [@mccauley11]. Like these previous models, Nümila reduces the problem of language acquisition to the much simpler problem of producing grammatical utterances based on statistical patterns in speech, specifically the transitional probabilities between words and phrases. In reality, language acquisition is heavily dependent on the semantic and social aspects of language [@tomasello03; @goldstein10], aspects which the present model does not capture. However it is generally agreed that linguistic pattern recognition plays at least some role in language acquisition; thus, the present model can be seen as a baseline that could be improved upon by enriching the input to the model with environmental cues.

The model is theoretically aligned with usage-based psycholinguistics, a field that emphasizes psychological plausibility and empirically observed behavior. The model is highly incremental, processing one utterance at a time. For each utterance, the model applies a parsing algorithm that simultaneously assigns structure to and learns from the utterance. Producing utterances relies on the same basic principles as parsing an utterance, and the representations underlying all processing take the same form as the main knowledge base.

In line with many usage-based psycholinguistic models, the present model is fairly simplistic [although see @bannard09]. It does not explicitly represent abstractions such as syntactic categories and dependencies. However, this is a characteristic of the particular model, and not of graphical models in general. Although we suggest the possibility that rule-like behavior could emerge from the present model without explicit rule-like representations, we do not make any strong theoretical claims about human linguistic representations.


## Graphical model
The model represents its knowledge using a BindGraph. Words and phrases ("chunks") are stored as nodes, and transitional probabilities between those elements as edges. The full graph has the same structure as a parse of a single utterance, as shown in figure \ref{fig:graph}.

\input{diagrams/graph.tex}

### Edges
The model has two edge-types representing forward and backward transitional probabilities, that is, the probability of one word following or preceding a given word: $p(w_i = x | w_{i-1} = y)$ and $p(w_i = x | w_{i+1} = y)$ respectively. Although forward transitional probability (FTP) is the standard in N-gram models, some evidence suggests that infants are more sensitive to BTP [@pelucchi09], and previous language acquisition models have been more successful when employing it [@mccauley11]. To examine the relative contribution of each direction of transitional probability, we make their relative weight an adjustable parameter. Although ADIOS and U-MILA have only one type of temporal edge (co-occurrence count), their learning algorithms compute something very similar to FTP and BTP. By using two edge types, we build this computation into the representational machinery.


### Bind
When two nodes (initially words) are determined to co-occur at an unexpectedly high frequency (see below), the graph's bind function is applied to create a new node. As discussed above, the bind function, $fbind$, is a parameter of the graph, and must be a function from a list of nodes to a single node. As a simplifying assumption, we follow U-MILA by only considering binary binds; thus the function is of type $N \times N \rightarrow N$. Importantly, we do not make the theoretical claim that linguistic composition must be binary [as others have, c.f. @everaert15]; this decision was made for ease of modeling. We implement two such bind functions, one hierarchical and the other flat. Given arguments \n{[A B]} and \n{[C D]}, hierarchical bind returns \n{[[A B] [C D]]}, whereas flat bind returns \n{[A B C D]}. Both hierarchical [@solan05] and flat [@kolodny15; @mccauley11] merge rules have been used in previous models.

In the simplest case, the bind function determines only the identity of the new node. This quality alone has important ramifications. Hierarchical bind is a bijective function; that is, there is a one-to-one mapping from inputs to outputs. Conversely, flat bind is not bijective because multiple inputs can produce the same output. For example, if \n{[A B C]} and \n{D} occur together frequently, a new node \n{[A B C D]} will be created and added to the graph. Later on, if \n{[A B]} and \n{[C D]} are bound, we will get the existing node, \n{[A B C D]} with all its learned edge weights. In more practical terms, a model using a flat bind rule will treat every instance of a given string as the same entity. Although a flat binding rule will be unable to capture the semantic impact of constituent structure, it is unclear whether hierarchical information will be useful for the simple tasks we subject our model to. A potential downside of hierarchy is that it increases the number of possible structures a sentence can take, and thus increases the dimensionality of the search space for the learning algorithm.

We do not present results for the composition algorithm presented above due to computational constraints. However, we note that exploratory simulations indicated that it did not improve performance on the natural language corpora.

## Learning

The model constructs a BindGraph given an initially blank slate by processing each utterance, one by one. Thus, the model has more limited memory resources than both ADIOS and U-MILA. The graph is constructed with three operations: (1) adding newly discovered base tokens to the graph, (2) increasing weights between nodes in the graph, and (3) creating new nodes by binding existing nodes. We implement two processing algorithms that employ these basic operations to learn from an utterance. The first is meant to replicate U-MILA's bottom up chunking mechanism, learning transitional probabilities between all possible nodes in the utterance. The second is inspired by the Now-or-Never bottleneck [@christiansen15], incorporating an even more severe memory constraint, and building up a structural interpretation of the utterance word by word.

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


### GreedyParse

The GreedyParse algorithm follows the same basic principles as FullParse in that it is based on updating transitional probability edges and binding nodes. However, unlike FullParse, GreedyParse incorporates severe memory limitations and online processing restraints. In contrast to FullParse, which finds all possible structures for an utterance given the current graph, GreedyParse finds a single structure by making a sequence of locally optimal decisions, hence "Greedy". Upon receiving each word it can create at most one chunk and the nodes used in this chunk can not be used later in a different chunk. Thus, the algorithm may not assign the optimal structure to the utterance.

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




# Testing the model on natural language
To test the model, we use naturalistic child directed speech, specifically the corpora prepared by @phillips14. For each of the seven languages, the input can be tokenized by word, syllable, or phoneme, giving a total of $7 \times 3 = 21$ corpora. All models are trained on the first 7000 utterances of each corpus, and tested on the next 1000. We test several instantiations of Nümila using different BindGraph implementations, parsing algorithms, and bind functions.

## Experiment 1: Grammaticality judgment
As a first test, we use the common task of discriminating grammatical from ungrammatical utterances. This task is appealing because it is theory agnostic (unlike evaluating tree structures) and it does not require that the model produce normalized probabilities (unlike perplexity). The only requirement is that the model be able to quantify how well an utterance fits its knowledge of the language.

### Generating an acceptability score
Statistical language modeling is sometimes equated with determining the probability of word sequences [c.f. @goodman01], something that Nümila does not do natively because outgoing edges for one node (labeled e.g. FTP) are not required to sum to 1, and are thus not probabilities. Of course, it would be possible to calculate normalized edge probabilities; however, we take the alternative approach of evaluating the model with the model-neutral task of grammaticality discrimination.

Although Nümila's edge weights are not probabilities, we employ the same basic principles of formal language models to assign a score. In a generative language model, the probability of an utterance is the sum of the probabilities of all possible ways to produce the utterance (e.g. all tree structures). The probability of each structure is the product of the probabilities of every rule that is applied in creating the utterance. With a PCFG, the rules are all of the form $NT \rightarrow \alpha$, where $NT$ is a nonterminal symbol (representing one branch of the tree, a constituent) and $\alpha$ is a sequence of symbols, either terminal or nonterminal. With an N-gram model, on the other hand, the rules are all of the form $\alpha \rightarrow \alpha \cdot w$, where $\alpha$ is the $N-1$ most recent words and $w$ is the next word.

Because Nümila incorporates structural elements (chunks) and transitional probabilities, it uses both types of rules. For chunks, the PCFG rule is applied; however, because each node has exactly one compositional structure, the rule probability is always 1. When an utterance has a series of nodes that cannot be combined, the bigram rule is applied: For each adjacent pair of nodes, $(X, Y)$, we apply the rule $X \rightarrow X \cdot Y$ with probability proportional to $E_F(X, Y)$. The result is simply the product of FTPS between each pair of nodes spanning the utterance. Finally, to avoid penalizing longer utterances, we take the result to the $n-1$ root where $n$ is the length of the utterance. We choose this number because it is the maximum number of edge weights that could be multiplied together to produce the final score.

Given a function that assigns scores to a single parse of an utterance, it is straightforward to create a function that assigns scores to the utterance itself. With a PCFG (where the scores are probabilities) the probability of the utterance is the sum of the probabilities of each parse. This is a result of the assumption that the utterance is generated by the PCFG model, along with Kolmogorov's third axiom that the probability of the union of independent events is the sum of the probabilities of each event. Although Nümila's scores are not true probabilities, we apply the same rule. That is, the score of an utterance is the sum of the scores for all parses of that utterance.

### Preparation of stimuli and analysis of performance
To construct a test corpus, we first take 500 unseen utterances from the corpus, which are labeled "grammatical". For each utterance, we create a set of altered utterances, each with one adjacent pair of tokens swapped. For example, given "the big dog", we create "big the dog" and "the dog big". These altered utterances are added to the test corpus with the label "ungrammatical". The models task is to separate grammatical from ungrammatical. Often, this task is modeled by setting a threshold, all utterances being predicted to be grammatical. However, it is unclear how to set such a threshold without either specifying it arbitrarily or giving the model access to the test labels. Thus, we employ a metric from signal detection theory, the Receiver Operator Characteristic.

![A Receiver Operator Characteristic curve for two models on the English word corpus.](figs/roc-curve.pdf){#fig:roc-curve}

An ROC curve, such as the one in figure @fig:roc-curve, plots true positive rate against false positive rate. As the acceptability threshold is lowered, both values increase. With random scores, they will increase at the same rate, resulting in a line at $y=x$. A model that captures some regularities in the data, however, will initially have a faster increasing true positive rate than a false positive rate because the high-scored utterances will tend to be grammatical ones. This results in a higher total area under the curve, a scalar metric that is often used as a metric of the power of a binary classifier. This measure is closely related to precision and recall, but has the benefit of allowing interpolation between data points, resulting in a smoother curve [@davis06].

### Results

Overall, Nümila preforms better than chance, but no better than a bigram model. The various instantiations of Nümila all preform roughly the same, although the noise introduced by the HoloGraph has a slight negative impact. There does not appear to be any interaction with language or input type with the notable exception that the non-hierarchical model using a greedy parsing algorithm performs poorly with the phonetic input. We have no explanation for this anomaly.

![Area under ROC curve for different languages, collapsed across input type.](figs/roc-lang.pdf){#fig:roc-lang}

![Area under ROC curve for different input types, collapsed across languages.](figs/roc-type.pdf){#fig:roc-type}

## Experiment 2: Production
As a second test, we use the task of ordering a bag of words---a proxy for production. A more direct test of production would be to generate utterances without any input, for example, by concatenating nodes in the graph based on transitional probabilities. However, this task has two disadvantages. First, it is difficult to evaluate the acceptability of generated utterances without querying human subjects. Second, utterance production in humans likely involves semantic as well as structural information, the first of which the present model does not attempt to capture. To avoid these problems, we follow previous work [@chang08; @mccauley14a] by using a word-ordering task to isolate structural knowledge. A bag of words is taken as an approximate representation of the thought a speaker wishes to convey; the syntactic task is to say the words in the right order.

### Ordering a bag of words
We treat ordering a bag of words as an optimization problem, using the acceptability score described above as a utility function. The optimal but inefficient strategy is to enumerate all possible orderings of the words and choose the one with the highest acceptability score. However, with $n!$ possible orderings, this becomes intractable for longer utterances. As with learning, we propose a greedy algorithm to approximate the optimal solution. Typically, such an algorithm starts from the beginning of the utterance and iteratively appends the best word or chunk to the end, producing the utterance in the same order it was spoken [e.g. @mccauley14a; @kolodny15]. Incremental production has some theoretical appeal given that words are ultimately produced in this order. However, it is not clear that utterances are planned in this way. Lacking strong theoretical motivation for purely incremental sentence planning, we explore a more flexible approach.

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
To test the model on this task, we take an unseen item from the corpus, convert it into a bag of words, and then compare the model's ordering to the original utterance. A simple comparison strategy is to assign a score of 1 if the model's output perfectly matches the original, and 0 otherwise [as in @mccauley14a]. However, this metric selectively lowers the average score of longer utterances, which have $n!$ possible orderings. If the average score varies across utterance lengths, utterances of different lengths will have varying discrimination power (in the extreme, no discrimination power if all models fail all utterances of a given length). Given this, we use the BLEU metric [@papineni02], which is more agnostic to utterance length. Specifically, we use the percentage of bigrams that are shared between the two utterances.

### Results
The production results are more discriminative than the grammaticality results.

Nümila outperforms the bigram model. However, the speaking algorithm was developed with chunks in mind, so this might be an unfair comparison.

![Production results.](figs/bleu.pdf)

- worse performance of fullparse due to not focusing on a particular path for constructing chunks, part of speaking algo


