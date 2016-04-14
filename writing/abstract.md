
What representations underlie language knowledge and use? For an
empirically motivated or usage-based model to be viable, the
representation it posits must be both psychologically plausible and
computationally powerful enough to capture the complexity of
language. @solan05 proposed a representation in the form of a
structured graph over elementary units such as phonemes or words and
demonstrated that the resulting model can perform some of the tasks
arising in language acquisition and use. Graphs are appealing in this
setting because they are easy to interpret, domain-general, and
capable of supporting a variety of language-related phenomena. The
question of how best to map graph-like representations onto brain
circuits remains, however, open. To address it, we constructed an
approximate, distributed representation of a graph on top of a
neurally plausible vector space. This representation serves as the
basis for the development of a novel usage-based online language
acquisition model. Our results suggest that graphs can be effectively
represented in a neurally plausible manner and used to support
language acquisition and processing.

@solan05 recognized that graphs are well suited to representing both
statistical patterns and structural rules. In a simple case, a bigram
model can be represented as a graph with words as nodes and
transitional probabilities as edges. A multigraph (a graph with
multiple edge types) can represent both forward and backward
transitional probabilities, both of which can be useful in identifying
patterns in natural language.  Edges representing composition and
class membership can be used to capture the hierarchical and
categorical structure of language. The ADIOS model incorporated all
these edge types, allowing it to perform surprisingly well (for a
model based on unsupervised learning) in a range of statistics- and
structure-related tasks.

However, ADIOS employed a psychologically implausible batch learning
algorithm. @kolodny15 aimed to address this concern by developing an
online process model of sequential structure learning, using a similar
graph-based representation. Unlike ADIOS, which allows multiple passes
over the full corpus (incorporated from the outset into the graph),
their model, U-MILA, constructs the graph incrementally, as it
receives more and more utterances. U-MILA was shown to replicate a
number of psycholinguistic results in word segementation, category
learning, and structural dependency learning.

Given that a graph-like representation of language knowledge can be
constructed online in a psychologically plausible manner, we may ask
further how the requisite graph could be implemented in the brain. At
one level, the brain literally *is* a graph, with neurons as nodes and
synapses as edges. It is, however, implausible that words or phrases
(nodes in the above models) would be represented by single
neurons. More likely, there is a layer of abstraction between the
level of the symbolic graph and the level of neurons. Specifically,
nodes and edges in a graph are likely represented in a distributed
manner by ensembles of neurons. Thus, what we need is a link between a
symbolic graph and a distributed neural representation.

Vector space models provide just such a link. A symbolic node (and its
edges) can be represented as a point in a high dimensional
space. Batch-computed vector space representations have been widely
used in approaches such as Latent Semantic Analysis [@deerwester90]
and Topic Models [@griffiths07]. The BEAGLE model emloys holographic reduced representations [@plate95] to create vector representations of words online [@jones07]. However, the present model is the first to our knowledge that employs holographic representations in a productive language acquistion model. 

To represent a directed graph in a holographic vector space, we
replace the traditional N x N adjacency matrix with an N x D matrix
where D is the dimensionality of our sparse vectors. Every time a node
is added to the graph, it is assigned a random id-vector and a new row
in the matrix. The row represents the outgoing edges from this
node. To increase the edge weight from node A to node B, we simply add
B's id-vector to node A's row-vector. Intuitively, the more frequently
this operation occurs, the more A's row-vector will be determined by
B's id-vector. Thus, we can measure the weight of the edge from A to B
by taking the cosine similarity of A's row-vector and B's
id-vector. Furthermore, each time B's id-vector is added to A's
row-vector, the similarity of all other id-vectors to A's row-vector
decreases very slightly. Thus, edge weights behave somewhat like
probabilities (although they neeed not sum to exactly one).

To demonstrate that a graph represented in this manner can be useful
for language modeling, we implement a simplified graph-based language
acquisition model based on ADIOS and U-MILA. Our model incrementally constructs a graph of words and *chunks*, the latter being identified by high forward and backward transitional probabilities.
Despite a strict memory limitation of four nodes, the model can represent longer sequences through hierarchical chunking: a six-word clause may be
represented by a single node. These chunks are built up
from their constituents using a binding operation such as circular
convolution [@plate95] or random permutation [@recchia15].

NOTES:
- You wrap your paragraphs by hand? Why not turn on line wrapping in your editor? It's so much easier! 
- The models used for the graphs actually don't use binding becaues it hinders performance
- The previous sentence clashes with next one. In reality, the graphs have different binding operation that function quite differently. I do hope to unify these at some point.
- We could just throw the binding talk out, but it is one of the more interesting things about vectors.

Because the acquisition model is indifferent to how its graph is
implemented, we instantiate two forms of the model, one with a
vector-based holographic graph, and one with a traditional
adjacency-matrix graph (normalizing outgoing edges to
make them probabilities). Our results show that the model performs
comparably with the two graph instantiations, suggesting that the
holographic graph functions similarly to a standard graph in the
present case. This preliminary evidence suggests that graphs may
indeed have a feasible neural implementation, making them a powerful
tool for cognitive modelers in any domain, including language.

## Results

We first assess our model by its ability to discriminate adult utterances from ungrammatical word sequences. Figure 1 shows the receiver operating characteristic of our model and an SRILM bigram model with additive smoothing (the best preforming NGram model we tried).

We model production as the task of ordering a bag of words, a task meant to isolate structural knowledge [@chang08; @mccauely14a]. The model preforms this task by incrementally combining words into chunks until a single chunk emerges as the produced utterance. We evaluate the output using a bigram BLEU score: the percentage of bigrams shared between the original and model-ordered adult utterance. Figure 2 shows that the model outperform the bigram model

## References
