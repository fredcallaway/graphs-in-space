# Nümila
To demonstrate that a VectorGraph can be used as a component of another model, we created a simple language acquisition model, based on previous graphical models [@solan05; @kolodny15] and chunking models [@mccauley11]. Like these previous models, Nümila reduces the problem of language acquisition to the much simpler problem of producing grammatical utterances based on statistical patterns in speech, specifically the transitional probabilities between words and phrases. In reality, language acquisition is heavily dependent on the semantic and social aspects of language [@tomasello03; @goldstein10], perhaps so much so that it cannot be effectively studied without considering these factors [see @frank09]. Thus, we present the model mainly for illustrative purposes.

Nümila is a hybrid of ADIOS, U-MILA, and the Chunk Based Learner. It has the hierarchical representations of ADIOS, the bottom-up learning algorithm of U-MILA (roughly), and the incrementality of CBL. Nümila consists of a graphical model, a parsing algorithm, and a production algorithm. 


## Graphical model
The model represents its knowledge using a directed, labeled multigraph. Because the model requires only the basic operations of bumping and retrieving edge weights, any graph that implements these operations, with at least pseudo-probabalistic edge weights can be used. We test realizations of the model using a VectorGraph and a ProbGraph, as described above. Words and phrases ("chunks") are nodes, and transitional probabilities between those elements are edges. An idealized visualization of the graph is shown in Figure 5.  

\input{diagrams/graph.tex}

### Edges
The model has two edge-types representing forward and backward transitional probabilities, that is, the probability of one word following or preceding a given word: $p(w_i = x | w_{i-1} = y)$ and $p(w_i = x | w_{i+1} = y)$ respectively. Both edge weights are used when computing the _chunkiness_ of two nodes. Chunkiness measures the degree to which two words tend to occur together, and is defined as the weighted geometric mean of FTP and BTP between the two nodes. The chunkiness for the ordered pair $(a, b)$ is

$$ \text{chunkiness}(a, b) = \sqrt{
    w_{\n{FTP}} \weight(a, b, \n{FTP}) \ 
    w_{\n{BTP}} \weight(b, a, \n{BTP}) } $$ {#eq:chunkiness}


Although forward transitional probability (FTP) is the standard in N-gram models, some evidence suggests that infants are more sensitive to BTP [@pelucchi09], and previous language acquisition models have been more successful when employing it [@mccauley11]. To examine the relative contribution of each direction of transitional probability, we make their relative weight an adjustable parameter.


### Merge
When two nodes (initially words) are determined to co-occur at an unexpectedly high frequency (see below), a merge function is applied to create a new node. We consider three merge functions. The _flat_ merge function takes two[^binary] nodes and concatenates them: \n{[A B], [C D]} $\rightarrow$ \n{[A B C D]}. The _hierchical_ merge combines them into a tree: \n{[A B], [C D]} $\rightarrow$ \n{[ [A B] [C D] ]}. Finally, the _compositional_ merge is like hierarchical merge, but additionally uses the composition algorithm discussed above to construct initial edge weights for the newly created node.

[^binary]: The restriction to a binary merge function is a simplifying assumption, not a theoretical claim [in contrast to @chomsky99].


## Parsing
The assignment of structure and learning occur in a single process that we call parsing. To parse an utterance, the model constructs a path through the graph, making local modifications as it goes. The initially blank graph is thus built with three operations: (1) adding newly discovered base tokens to the graph, (2) increasing weights between nodes in the graph, and (3) creating chunk nodes by merging existing nodes. In addition to modifying the graph, parsing assigns structure to the utterance, representing it as a path through the graph. The path spans the full utterance; however, because chunk nodes span multiple words, this path may have fewer nodes than the utterance has words.

The model uses a greedy algorithm to construct the path. It begins at the boundary node, and then proceeds forward one word/node at a time. Whenever a new node is added to the path, the FTP and BTP edge weights between that node and the previous node are bumped. In line with the effect of working memory on language use [@christiansen15], the model operates in a narrow, four-node window. When the path becomes four nodes long, the model attempts to consolidate by replacing pairs of nodes with chunks. To do this, the chunkiness (Eq. @eq:chunkiness) between all adjacent nodes is computed; if the highest chunkiness exceeds a threshold, the pair is replaced by a single node representing that pair. If this chunk node is already in the graph, it is used; otherwise it is first created and added to the graph. Edge weights between this new node and the nodes on either side are bumped. If no pair exceeds the threshold, the oldest node in the path is removed from working memory to maintain the four-node window; it receives no additional processing.

The algorithm proceeds in this manner, adding the next word to the path and chunking, until it reaches the end of the utterance, at which point nodes are combined until the path consists of a single node, or no pair of nodes exceeds the chunkiness threshold. The final representation of the utterance is the full path, including those nodes dropped from working memory.


## Simulations
To test the model, we use naturalistic child directed speech, specifically the corpora prepared by @phillips14. For each of the seven languages, the input can be tokenized by word, syllable, or phoneme, giving a total of $7 \times 3 = 21$ corpora. All models are trained on the first 7000 utterances of each corpus, and tested on the next 1000. We test several instantiations of Nümila using different graph implementations, merge functions, and parameter settings. 

### Grammaticality judgment
As a first test, we use the common task of discriminating grammatical from ungrammatical utterances. This task is appealing because it is theory agnostic (unlike evaluating tree structures) and it does not require that the model produce normalized probabilities (unlike perplexity).

#### Generating an acceptability score
To score an utterance, the model begins by parsing the utterance, discovering a path through the graph that passes through every word in the sentence (possibly visiting multiple words with one chunk node). The product of chunkinesses for every adjacent pair of nodes on the path is then calculated. Finally, to avoid penalizing longer utterances, the score is taken to the $n-1$ root, where $n$ is the length of the utterance. 


#### Preparation of stimuli and analysis of performance
To construct a test corpus, we first take 500 unseen utterances from the corpus, which are labeled "grammatical". For each utterance, we create a set of altered utterances, each with one adjacent pair of tokens swapped. For example, given "the big dog", we create "big the dog" and "the dog big". These altered utterances are added to the test corpus with the label "ungrammatical". The model's task is to separate grammatical from ungrammatical. Often, this task is modeled by setting a threshold, all utterances with scores higher than the threshold predicted to be grammatical. However, it is unclear how to set such a threshold without either specifying it arbitrarily, or giving the model access to the test labels. Thus, we employ a metric from signal detection theory, the Receiver Operator Characteristic.

![A Receiver Operator Characteristic curve showing the tradeoff between true- and false-positive rate on a grammaticality discrimination task. The default Nümila model and a random, dummy model are shown.](figs/roc-curve.pdf){#fig:roc-curve width=70%}

An ROC curve, such as the one in figure @fig:roc-curve, plots true positive rate against false positive rate. As the acceptability threshold is lowered, both values increase. This curve is closely related to the precision-recall curve, but it has the benefit of allowing interpolation between data points, resulting in a smoother curve [@davis06]. As a scalar metric, we use the total area under the curve. The better the separation of grammatical from ungrammatical by the acceptability score, the higher this value will be.

#### Results

The results indicate that the VectorGraph can be successfully used in a language acquisition model. The models using a VectorGraph perform about as well as those using a ProbGraph, which has true transitional probabilities as edges. However, they perform slightly worse in some cases, likely due to interference between non-orthogonal index vectors (as in @fig:storage). To support this explanation, we note that the Vector/Prob difference is most pronounced for words, of which there are the most unique tokens. Looking only at the phonemic input, we see that the two graphs perform more similarly.

Using multi-word chunks does not provide any benefit---in fact, it seems to be a disadvantage. The Markovian model that only tracks transitional probabilities (both FTP and BTP) between individual words does the best[^SRILM]. There is no clear difference between the hierarchical and flat merging rules, but given that chunking is not working at all, we cannot infer much from this result. However, we note that our method of producing foil utterances may be especially well-suited for a first-order Markov model. By simply swapping adjacent words, we are likely to introduce unlikely bigrams, which the Markov model will capture. Thus, we turn now to the next simulation, which may better distinguish the models.

[^SRILM]: We initially attempted to use SRILM bigram and trigram models as a baseline. We found that, with a variety of parameter settings, they consistently performed worse than a the FTP-only, non-chunking Nümila (a bigram model). We suspect this is due to optimizations designed to work for larger corpora.

![Grammaticality discrimination. Area under ROC curve for different input types and models, collapsed across languages. Error bars here and in all subsequent plots represent 95% confidence intervals by bootstrapping. Vector/prob refer to the graph. Flat indicates that the non-hierarchical merge rule is used. Markov indicates that chunks are not used. Comp indicates the compositional merge function is used.](figs/roc-type.pdf){#fig:roc-type}

### Production
As a second test, we use the task of ordering a bag of words, a proxy for production. A more direct test of production would be to generate utterances without any input, for example, by concatenating arbitrary nodes in the graph based on transitional probabilities. However, this task has two disadvantages: (1) it is difficult to evaluate the acceptability of generated utterances without querying human subjects, and (2) speaking involves semantic as well as structural information, the first of which the present model does not attempt to capture. To avoid these problems, we follow previous work [@chang08; @mccauley14a] by using a word-ordering task to isolate structural knowledge. A bag of words is taken as an approximate representation of the thought a speaker wishes to convey; the syntactic task is to say the words in the right order.

#### Ordering a bag of words
We treat ordering a bag of words as an optimization problem, using the acceptability score described above as a utility function. The optimal but inefficient strategy is to enumerate all possible orderings of the words and choose the one with the highest acceptability score. However, with $n!$ possible orderings, this becomes intractable for longer utterances. As with parsing, we propose a greedy algorithm, very similar to the one used by @mccauley15. As with parsing, production can be seen as forging a path through the graph; however in this case, the model must choose the order in which it visits the nodes.

The algorithm begins by greedily constructing chunks using the input words: The words are placed in a bag, and the most chunkable pair is replaced with their chunk node until no more chunks can be made. Because a word and chunk are both nodes, this process is applied recursively, sometimes resulting in a single chunk for the entire utterance. Next, the chunks are combined to form an utterance: Beginning from the utterance boundary node, the node in the that has the greatest chunkiness with the previous node is added to the path. This is somewhat like a Markov process, except that chunkiness is used in the place of forward transitional probability and a maximization rule is used as opposed to probability matching.


#### Preparation of stimuli and analysis of performance
To test the model on this task, we take an unseen item from the corpus, convert it into a bag of words, and then compare the model's ordering to the original utterance. A simple comparison strategy is to assign a score of 1 if the model's output perfectly matches the original, and 0 otherwise [as in @mccauley14a]. However, this metric selectively lowers the average score of longer utterances, which have $n!$ possible orderings. If the average score varies across utterance lengths, utterances of different lengths will have varying discrimination power (in the extreme, no discrimination power if all models fail all utterances of a given length). Given this, we use the BLEU metric [@papineni02], which is more agnostic to utterance length. Specifically, we use the percentage of trigrams that are shared between the two utterances.[^bleu]

[^bleu]: Although we present only results with this metric, we remark that the pattern of results is roughly constant for various order N-grams, as well as for the strict perfect match metric.

#### Results
The production results roughly parallel the grammaticality discrimination results. The VectorGraph suffers slightly from noise, and the Markovian models perform the best. In this case, the nature of the task does not explain the results. CBL outperforms trigrams on a very similar task, indicating that chunks can be useful for this task [@mccauley15]. We remark, however, that their model only attempts to produce the utterances spoken by the child, while our model attempts to produce adult-generated utterances. If chunks play a more significant role in the linguistic knowledge of a child than that of an adult, we would predict a chunking model to perform better on child-produced utterances.


![Production. The percentage of shared trigrams [i.e. third-order BLEU score; @papineni02] between an adult utterance and the model's attempt to construct an utterance given an unorderd bag-of-words representation of that utterance.](figs/bleu.pdf){#fig:bleu}

To  investigate the contribution of FTP and BTP, we conduct the production experiment with a second set of models. All use the ProbGraph, to eliminate possible interactions with the noise caused by the VectorGraph. We use a 2 $\times$ 2 design of (chunking, Markov) $\times$ (ftp, btp, both). Results are displayed in @fig:tp-bleu.  For the Markovian models, there is a clear preference for BTP, which is somewhat surprising given that the total score of a path is the same regardless of transition direction. This indicates that the FTP/BTP distinction may depend on the greedy production algorithm, which chooses the next word based on the transition from the last word. A possible explanation is that BTP prevents the algorithm from immediately concatenating a common word (which will generally have higher incoming FTP). When using BTP, the frequency of a word will not affect its probability of being chosen (at least, not in this way).

Another surprising result is the interaction between chunking and transitional direction. The chunking models are not as sensitive to this attribute: The chunking model outperforms the Markov model when only using FTP. Given that the learning of chunks, as well as the chunking phase of production, operate entirely based on FTP, we are unsure how to explain these results. Additionally, we are unable to dissociate the effect of transitional probability direction on the learning and production phases.

![Relative contribution of FTP and BTP.](figs/tp-bleu.pdf){#fig:tp-bleu}

## Discussion
The performance of Nümila is less than impressive. However there are still important lessons to be learned. First, and most relevant here, is the observation that the VectorGraph can be used effectively in language acquisition model with naturalistic input. Unfortunately, the fact that chunking does not seem to improve performance limits our ability to judge the effectiveness of the composition algorithm. The non-effect of the predictive merge function could be due to its inadequacy, or to the ineffective learning and use of chunks in general.

We may have been able to identify an effect of chunking with easier tasks. For example, we could follow @mccauley15 by giving the model child-produced utterances in the bag-of-words task. We could also follow @bod09 by using parts of speech as input. Finally, given the relatively small inventory of phonemes compared to words and syllables, we may be able to tease out an effect of chunking with a word-segmentation task.

