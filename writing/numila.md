# Nümila
To demonstrate that a VectorGraph can be used as a component of another model, we created a simple language acquisition model, based on previous graphical models [@solan05; @kolodny15] and chunking models [@mccauley11]. Like these previous models, Nümila reduces the problem of language acquisition to the much simpler problem of producing grammatical utterances based on statistical patterns in speech, specifically the transitional probabilities between words and phrases. In reality, language acquisition is heavily dependent on the semantic and social aspects of language [@tomasello03; @goldstein10], perhaps so much so that it cannot be effectively studied without considering these factors [see @frank09]. Thus, we present the model mainly for illustrative purposes, rather than as a proposed model of language acquisition.

Nümila is a hybrid of ADIOS, U-MILA, and CBL. It has the hierarchical representations of ADIOS, the bottom-up learning algorithm of U-MILA (roughly), and the incrementality of CBL. Nümila consists of a graphical model,a parsing algorithm, and a production algorithm. 


## Graphical model
The model represents its knowledge using a directed, labeled, multigraph, such as the VectorGraph. Words and phrases ("chunks") are nodes, and transitional probabilities between those elements are edges. An idealized visualization of the graph is shown in \ref{fig:graph}.

\input{diagrams/graph.tex}

### Edges
The model has two edge-types representing forward and backward transitional probabilities, that is, the probability of one word following or preceding a given word: $p(w_i = x | w_{i-1} = y)$ and $p(w_i = x | w_{i+1} = y)$ respectively. Although forward transitional probability (FTP) is the standard in N-gram models, some evidence suggests that infants are more sensitive to BTP [@pelucchi09], and previous language acquisition models have been more successful when employing it [@mccauley11]. To examine the relative contribution of each direction of transitional probability, we make their relative weight an adjustable parameter. Although ADIOS and U-MILA have only one type of temporal edge (co-occurrence count), their learning algorithms compute something very similar to FTP and BTP. By using two edge types, we build this computation into the representational machinery.


### Merge
When two nodes (initially words) are determined to co-occur at an unexpectedly high frequency (see below), a merge function is applied to create a new node. We consider three merge functions. In Nümila, a merge function has two purposes: (1) to determine the identity of the resulting node, and optionally (2) to construct initial edge weights for the node. The _flat_ merge function takes two[^binary] nodes and concatenates them, e.g. \n{[A B], [C D]} $\rightarrow$ \n{[A B C D]}. The _hierchical_ merge combines them in a tuple, e.g. \n{[A B], [C D]} $\rightarrow$ \n{[ [A B] [C D] ]}. Finally, the _compositional_ merge is like hierarchical merge, but additionally uses the composition algorithm discussed above to construct initial edge weights for the newly created node.

[^binary]: The restriction to a binary merge function is a simplifying assumption, not a theoretical claim [in contrast to @chomsky99].


## Parsing
The assignment of structure and learning occur in a single process that we call parsing. To parse an utterance, the model constructs a path through the graph, making local modifications as it goes. The initially blank graph is thus built with three operations: (1) adding newly discovered base tokens to the graph, (2) increasing weights between nodes in the graph, and (3) creating chunk nodes by merging existing nodes. In addition to modifying the graph, parsing assigns structure to he utterance, representing it as a path through the graph. The path spans the full utterance; however, because chunk nodes span multiple words, this path may have fewer nodes than the utterance has words.

The model uses greedy algorithm to construct the path. It begins at the boundary node, and then procedes forward one word/node at a time. Whenever a new node is added to the path, the FTP and BTP edge weights between that node and the previous node are bumped. In line with the Now-or-Never bottleneck [@christiansen15], the algorithm has a limited working memory: It can only "see" the four most recent nodes in the path. When this limit is reached, the algorithm attempt to consolidate the path by replacing two nodes with a single chunk node. To do this, the chunkiness between all adjacent nodes is computed; if the highest chunkiness exceeds a threshold, the pair is replaced by a single node representing that pair. If this chunk node is already in the graph, it is used; otherwire it is first created and added to the graph. Finally, the edge weights between this new node and the nodes on either side are bumped.

The algorithm procedes thusly, adding the next word to the path and chunking, until it reaches the end of the utterance. At this point, nodes are combined until the path consists of a single node, or no pair of nodes exceeds the chunkiness threshold. The full path, including nodes that were dropped from memory, are retained in the final representation of the parse.


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


## Simulations
To test the model, we use naturalistic child directed speech, specifically the corpora prepared by @phillips14. For each of the seven languages, the input can be tokenized by word, syllable, or phoneme, giving a total of $7 \times 3 = 21$ corpora. All models are trained on the first 7000 utterances of each corpus, and tested on the next 1000. We test several instantiations of Nümila using different grapd implementations, \todo{parsing algorithms}, and merge functions.

## Experiment 1: Grammaticality judgment
As a first test, we use the common task of discriminating grammatical from ungrammatical utterances. This task is appealing because it is theory agnostic (unlike evaluating tree structures) and it does not require that the model produce normalized probabilities (unlike perplexity). The only requirement is that the model be able to quantify how well an utterance fits its knowledge of the language.

### Generating an acceptability score
To score an utterance, the model begins by parsing the utterance, discovering a path through the graph that passes through every word in the sentence (possibly visiting multiple words with one chunk node). The product of chunkinesses for every adjacent pair of nodes on the path is then calculated. Finally, to avoid penalizing longer utterances, the score is taken to the $n-1$ root, where $n$ is the length of the utterance. 


TODO APPENDIX?

Statistical language modeling is sometimes equated with determining the probability of word sequences [c.f. @goodman01], something that Nümila does not do natively because outgoing edges for one node (labeled e.g. FTP) are not required to sum to 1, and are thus not probabilities. Of course, it would be possible to calculate normalized edge probabilities; however, we take the alternative approach of evaluating the model with the model-neutral task of grammaticality discrimination.

Although Nümila's edge weights are not probabilities, we employ the same basic principles of formal language models to assign a score. In a generative language model, the probability of an utterance is the sum of the probabilities of all possible ways to produce the utterance (e.g. all tree structures). The probability of each structure is the product of the probabilities of every rule that is applied in creating the utterance. With a PCFG, the rules are all of the form $NT \rightarrow \alpha$, where $NT$ is a nonterminal symbol (representing one branch of the tree, a constituent) and $\alpha$ is a sequence of symbols, either terminal or nonterminal. With an N-gram model, on the other hand, the rules are all of the form $\alpha \rightarrow \alpha \cdot w$, where $\alpha$ is the $N-1$ most recent words and $w$ is the next word.

Because Nümila incorporates structural elements (chunks) and transitional probabilities, it uses both types of rules. For chunks, the PCFG rule is applied; however, because each node has exactly one compositional structure, the rule probability is always 1. When an utterance has a series of nodes that cannot be combined, the bigram rule is applied: For each adjacent pair of nodes, $(X, Y)$, we apply the rule $X \rightarrow X \cdot Y$ with probability proportional to $E_F(X, Y)$. The result is simply the product of FTPS between each pair of nodes spanning the utterance. Finally, to avoid penalizing longer utterances, we take the result to the $n-1$ root where $n$ is the length of the utterance. We choose this number because it is the maximum number of edge weights that could be multiplied together to produce the final score.

Given a function that assigns scores to a single parse of an utterance, it is straightforward to create a function that assigns scores to the utterance itself. With a PCFG (where the scores are probabilities) the probability of the utterance is the sum of the probabilities of each parse. This is a result of the assumption that the utterance is generated by the PCFG model, along with Kolmogorov's third axiom that the probability of the union of independent events is the sum of the probabilities of each event. Although Nümila's scores are not true probabilities, we apply the same rule. That is, the score of an utterance is the sum of the scores for all parses of that utterance.



### Preparation of stimuli and analysis of performance
To construct a test corpus, we first take 500 unseen utterances from the corpus, which are labeled "grammatical". For each utterance, we create a set of altered utterances, each with one adjacent pair of tokens swapped. For example, given "the big dog", we create "big the dog" and "the dog big". These altered utterances are added to the test corpus with the label "ungrammatical". The model's task is to separate grammatical from ungrammatical. Often, this task is modeled by setting a threshold, all utterances with scores higher than the threshold predicted to be grammatical. However, it is unclear how to set such a threshold without either specifying it arbitrarily, or giving the model access to the test labels. Thus, we employ a metric from signal detection theory, the Receiver Operator Characteristic.

![A Receiver Operator Characteristic curve for two models on the English word corpus.](figs/roc-curve.pdf){#fig:roc-curve}

An ROC curve, such as the one in figure @fig:roc-curve, plots true positive rate against false positive rate. As the acceptability threshold is lowered, both values increase. This curve is closely related to the precision-recall curve, but it has the benefit of allowing interpolation between data points, resulting in a smoother curve [@davis06]. As a scalar metric, we use the total area under the curve. The better the separation of grammatical from ungrammatical by the acceptability score, the higher this value will be.

### Results

Overall, Nümila preforms better than chance, but no better than a bigram model. The various instantiations of Nümila all preform roughly the same, although the noise introduced by the HoloGraph has a slight negative impact. There does not appear to be any interaction with language or input type with the notable exception that the non-hierarchical model using a greedy parsing algorithm performs poorly with phonetic input. We have no explanation for this anomaly.

![Area under ROC curve for different languages, collapsed across input type.](figs/roc-lang.pdf){#fig:roc-lang}

![Area under ROC curve for different input types, collapsed across languages.](figs/roc-type.pdf){#fig:roc-type}

## Experiment 2: Production
As a second test, we use the task of ordering a bag of words---a proxy for production. A more direct test of production would be to generate utterances without any input, for example, by concatenating nodes in the graph based on transitional probabilities. However, this task has two disadvantages: (1) it is difficult to evaluate the acceptability of generated utterances without querying human subjects; (2) speaking involves semantic as well as structural information, the first of which the present model does not attempt to capture. To avoid these problems, we follow previous work [@chang08; @mccauley14a] by using a word-ordering task to isolate structural knowledge. A bag of words is taken as an approximate representation of the thought a speaker wishes to convey; the syntactic task is to say the words in the right order.

### Ordering a bag of words
We treat ordering a bag of words as an optimization problem, using the acceptability score described above as a utility function. The optimal but inefficient strategy is to enumerate all possible orderings of the words and choose the one with the highest acceptability score. However, with $n!$ possible orderings, this becomes intractable for longer utterances. As with parsing, we propose a greedy algorithm, very similar to the one used by @mccauley14a. As with parsing, production can be seen as forging a path through the graph; however in this case, the model must choose the order in which it visits the nodes.

The algorithm begins by greedily constructing chunks using the input words: These words are placed in a bag, and the most chunkable pair is replaced with their chunk node until no more chunks can be made. Now the chunks are combined to form an utterance: Beginning from the utterance boundary node, the node in the that has the greatest chunkiness with the previous node is added to the path. This is somewhat like a Markov process, except that chunkiness is used in the place of forward transitional probability and a maximization rule is used as opposed to probability matching.



### Preparation of stimuli and analysis of performance
To test the model on this task, we take an unseen item from the corpus, convert it into a bag of words, and then compare the model's ordering to the original utterance. A simple comparison strategy is to assign a score of 1 if the model's output perfectly matches the original, and 0 otherwise [as in @mccauley14a]. However, this metric selectively lowers the average score of longer utterances, which have $n!$ possible orderings. If the average score varies across utterance lengths, utterances of different lengths will have varying discrimination power (in the extreme, no discrimination power if all models fail all utterances of a given length). Given this, we use the BLEU metric [@papineni02], which is more agnostic to utterance length. Specifically, we use the percentage of bigrams that are shared between the two utterances.

### Results
The production results are more discriminative than the grammaticality results.

Nümila outperforms the bigram model. However, the speaking algorithm was developed with chunks in mind, so this might be an unfair comparison.

![Production results.](figs/bleu.pdf)

- worse performance of fullparse due to not focusing on a particular path for constructing chunks, part of speaking algo


