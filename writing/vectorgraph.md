# VectorGraph
Thus far we have seen that the graph is a powerful and flexible tool for modeling structured representation. We have also seen that high dimensional vector spaces and a small set of operations may provide a connection between symbolic models and neurons. In this section, we unite the two frameworks, describing an implementation[^python] of a graph using a VSA. If abstract models in linguistics can indeed be represented with graphs, and VSAs can indeed be implemented by neurons, a VSA implementation of graphs could lead to the unification of all three of Marr's levels of analysis.

[^python]: The code for the models and simulations presented in this paper can be viewed at https://github.com/fredcallaway/NU-MILA

To construct a vector representation of a graph, we begin with the traditional adjacency matrix. Noting similarities between this matrix and the co-occurrence matrices employed by distributional semantic models, we adopt a VSA-based method that has been used effectively in distributional models: _random indexing_ [see @sahlgren05 for a review]. The resulting data structure closely mimics the behavior of an adjacency matrix representation when the vectors have sufficiently high dimensionality.


## Random indexing for distributional semantic models
Distributional semantic models (DSMs), such as HAL [@lund96], LSA [@landauer97], and Topic Models [@griffiths07], share with VSAs the notion that an item can be represented by a high dimensional vector. In DSMs, a word's meaning is approximated by the contexts in which it is used. The word-context associations are represented in a very large and sparse matrix, with one row for each word and one column for each document (or word, depending on how context is defined). To the extent that words with similar meaning occur in the same contexts, these words will have similar rows (similarity generally operationalized with cosine similarity).

The raw context vectors are generally too large and sparse to use effectively. To address this, distributional models employ some form of dimensionality reduction, such as singular value decomposition. An alternative technique, as suggested by @kanerva00, is  _random indexing_. Rather than constructing the full word by document matrix and then applying dimensionality reduction, this technique does dimensionality reduction online. Each context is assigned an immutable _index vector_ which in Kanerva's implementation are sparse ternary vectors. The meaning of a word is represented by a _context vector_ (a domain-specific term for memory vectors). This vector is the sum of the index vectors for every context the word occurs in. Random indexing has been found to produce similar results to LSA with SVD at a fraction of the computational cost [@karlgren01].

## Random indexing for graphs


: Definition of symbols

|      Symbol      |                 Meaning                 |
|------------------|-----------------------------------------|
| $a, b, x, y, n$  | A node                                  |
| $ab$             | A node composed of $a$ and $b$          |
| $e$              | An edge label                           |
| $\edge[e]{x}{y}$ | The edge from $x$ to $y$ with label $e$ |
| $\id_x$          | The index vector of node $x$            |
| $\row_x$         | The row vector of node $x$              |
| $\Pi_e(\cdot)$   | The permutation function for label $e$  |

Here, we generalize Kanerva's technique to represent any graph. A standard representation of a graph is an adjacency matrix, where each row/column represents the outgoing/incoming edges of a single node. Applying this interpretation to the co-occurrence matrix used in a word-word distributional semantic model such as HAL, we have a graph with words as nodes and co-occurrence counts as edge weights. Observing that (1) random indexing can represent a co-occurrence matrix, and (2) a co-occurrence matrix can be interpreted as a graph, we suggest that random indexing can be used to represent any graph.

Indeed, we can directly map elements of Kanerva's algorithm to elements of a graph. Just as each context receives an index vector, each node receives an index vector. Context vectors become _row vectors_, which represent all outgoing edges of a node. Just as context vectors are the sum of index vectors of all contexts that a given word occurs in, row vectors are the sum of the index vectors of all nodes that a given node points to. Like context vectors, row vectors are a form of memory vector. They can be constructed incrementally with a series of edge weight increases: To bump the edge $\edge{x}{y}$, we add $\id_y$ to $\row_x$.

To recover the weight of $\edge{x}{y}$, we use cosine similarity: $\weight(\edge{x}{y}) = \cos(\row_x, \id_y)$. Intuitively, this value will be higher if $\edge{x}{y}$ has been bumped many times, because this means that $\id_y$ has been added to $\row_x$ many times. We can also use cosine similarity to measure similarity between nodes: $\simil(x, y) = \cos(row_x, row_y)$. Nodes that have similar outgoing edge weights will be similar because their row vectors will contain many of the same id vectors. Importantly, because random vectors in a high dimensional space tend to be nearly orthogonal, row vectors for nodes that share no outgoing edges will have similarity close to 0.


![Total outgoing edge weight for a single node as a function of number of edges. Vectors are length 500.](figs/total-edges.pdf){#fig:total-edges width=60%}


An interesting attribute of this representation is that edge weights behave somewhat like probabilities. That is, bumping $\edge{x}{y}$ will slightly decrease the weight of $\edge{x}{z}$. Visually, adding $\id_y$ to $\row_x$ pulls $\row_x$ towards $\id_y$, and thus away from $\id_z$. However, unlike probabilities, there is no hard bound on the sum of all edge weights for a node. The total outgoing edge weight for a single node increases as the number of outgoing edges increase, but at a decelerating rate, as shown in @fig:total-edges.

### Edge labels
A critical feature of U-MILA and ADIOS is the incorporation of different types of edges to represent different types of relationships. This is implemented in the VectorGraph using the _label_ operation. Following @basile11, we use permutation as the label operation. A permutation function simply rearranges the elements of a vector, and it can be represented by another vector of the same dimensionality. Thus, each label is assigned a unique, immutable vector the first time it is used. The permutation function for a label $e$, represented by the vector $\bm{p}$ is

$$ \Pi_e (\bm{v}) = [\bm{v}_{\bm{p}_0}, \bm{v}_{\bm{p}_1}, \dots, \bm{v}_{\bm{p}_D}] $$ {#eq:permutation}

Given that $\id_y$ represents an edge to $y$, we can represent a labeled edge to $y$ as $\Pi_e (\id_y)$. Thus, to increase the edge from $x$ to $y$ with label $e$ ($\edge[e]{x}{y}$) we add $\Pi_e (\id_y)$ to $\row_x$. Similarly, to recover the weight of $\edge[e]{x}{y}$, we use $\cos(\Pi_e (\id_y), \row_x)$.

```include
operations.md
```


## Generalization
Storing information in a structured form allows an agent to inform her decisions with past experience. However, for this knowledge to be widely applicable in the natural world, it cannot be rigid. This is especially true in complex systems such as language, where the exact same situation is unlikely to occur twice. When attempting to understand and react to events in these domains, an agent must generalize based on experiences similar but not identical to the current situation. For example, upon seeing a new breed of dog, you would likely still recognize it as a dog, and thus avoid leaving your meal unattended in its reach.

### Previous approaches to generalization
Traditionally, psychology and linguistics has assumed that _categories_ are the driving force behind generalization [@pothos11]. Under this view, generalization is mediated by the application of discrete labels to groups of stimuli (as in exemplar models) or features (as in prototype models). For example, in the above example, you first identify the unfamiliar animal as a dog, and only then infer that it, like other dogs, is liable to snatch up your dinner. Furthermore, much of the work on generalization treats categorization as an end-goal, rather than simply a means for informing actions [@kruschke92; @ashby95; @nosofsky86; @anderson91]. Note a parallel to linguistics, where assigning a structure and denotational meaning to a speech act is assumed as the goal, rather than responding reasonably to the utterance [as discussed by @clark97].

However, explicit categorization is only one possible mechanism for generalization. An alternative approach, coming from the parallel distributed processing group [@rumelhart86], is to learn higher order patterns in the environment directly, without mediating variables. This approach avoids challenges that arise with explicit categorization such as deciding when a group of items counts as a category and deciding which of several overlapping categories should determine a given property of an item [@pothos11]. A downside to this approach, however, is that the representations and computations underlying generalization are relatively opaque to the modeler, limiting the explanatory power of these models [@griffiths10].

Fortunately, we do not need to choose between these two approaches. Graphs are capable of representing both types of models. Many probabilistic models of categorization employ graphical representations [@tenenbaum11], whereas artificial neural networks are themselves graphs. In addition to these extreme ends of the spectrum, the VectorGraph can take an intermediary approach. We suggest such an approach here. Like the PDP approaches, the proposed generalization algorithm does not employ explicit categories or any other latent variables. Like category approaches, the basic units of the algorithm are individual items (and their feature vectors). This gives us a balance between the flexibility of the PDP approach and the interpretability of the category approach.



### Generalization in vector space
We view generalization as a function from a raw representation of an item to a generalized representation of that item. The function may be applied, for example, before retrieving an edge weight or measuring the similarity between nodes. In line with both PDP and categorical approaches, we take as a starting point the assumption that if two items are similar in many ways, they might also be similar in other ways. In spatial terms, if two vectors point in similar directions, they pull towards each other, becoming even more similar. If the vector space is uniformly distributed, this will only result in noise. However, if there is structure to the space (i.e. areas with higher and lower density), this results in a fuzzy version of online clustering in which vectors drift towards heavily populated areas.

Formally, for a node $x$, we create a generalized row vector as the sum of the all other nodes' row vectors, weighted by the similarity of each node, $n$ to $x$. The generalized row vector for $x$ is thus

$$ \text{gen}(\row_x) = \sum_{n \in G} \row_n \simil(x, n) $$ {#eq:gen}

## Compositionality
One form of generalization is unique and significant enough to merit separate discussion. The classical principle of compositionality, often attributed to Frege, states that the meaning of an expression is a function of the meanings of its constituent expressions and the rules for combining them. The principle is most often discussed in linguistics; however, language of thought theories [@fodor75; @goodman14; @piantadosi11] suggest that compositionality may be a fundamental characteristic of other kinds of higher order reasoning [see also @werning12].

Indeed, these models may shed greater insight on the role of compositionality in cognition. Language is a unique system because it is defined by individuals' attempts to represent it. Perhaps the compositional nature of language is a result of the human tendency to represent structure in this way. What role does compositionality play in other natural systems? Taking an example from @goodman14, an agent hoping to predict the outcome of a ping pong match might do so by composing the results of previous matches (blue beats green, green beats red) using probabilistic rules that describe the system (X beats Y $\wedge$ Y beats Z $\Rightarrow$ X beats Z). Importantly, the agent's internal model generally does not perfectly describe the system, but it still leads to useful predictions.

This example points to an important distinction between compositionality as a property of a system (e.g. ping pong tournaments) and compositionality as a tool that cognitive agents use to understand that system. The first is a topic for philosophers, and perhaps physicists, and there may be deep, absolute truths regarding this kind of compositionality. The second, more relevant to cognitive scientists, is not a formal property, but rather a tool an agent may use to predict the properties of some new element based on past experience with related elements. In this sense, compositionality is a form of generalization.

### Three approaches to compositionality in vector space
An immediate observation is that a bind operation such as circular convolution only makes a small step towards compositionality in the sense described above. The major challenge is how to encode the "rules" or patterns of compositionality. We see three possible approaches. The first is to create separate merge functions for separate domains. That is, the compositionality is encoded into the function itself. This appears to be the dominant strategy, exemplified by @plate95, who suggests many different ways to compose vectors. A drawback to this approach however, is that it requires knowledge of compositionality to take a fundamentally different form than other kinds of knowledge. This detracts from a major appeal of VSAs: the relatively transparent relation to neural processing. Additionally, it is likely that compositionality is itself generalized across domains to some extent. It's unclear how this could be done when the architecture of the merge function differs.

The second possible approach addresses this problem by representing the rules of compositionality numerically. For example, the compositionality of a given system could be represented as a vector which is bound to each input vector before combining them (through a binding or bundling operation). Alternatively, compositionality could be spread across vectors, each of which is used to label a given input vector with a particular role (e.g. agent, action, patient). These vectors will of course have to be learned, which will be a significant challenge. It may be that the amount of information needed to represent compositionality will be too great for first order vectors, thus a matrix (or higher order tensor) might be required. Indeed, whereas a vector maps onto the activations of an ensemble of neurons, a matrix maps onto the synapses between two ensembles. The second may be a more likely way to represent compositional patterns, given that compositionality is generally learned gradually.

In the third approach, compositionality is not represented separately from the individual items. Rather, the way that an item combines with other items is stored directly in that item, reminiscent of combinatory categorial grammar [@steedman11]. In this approach, the merge function itself could be very simple, perhaps just a bind operation. An advantage of the approach is that it treats a word's compositional behavior as no different from its other attributes, a theoretically elegant and perhaps intuitively appealing notion (e.g. adjective-iness is a feature of "red"). However, by forcing compositional features to reside in the same space as other features, the learning problem may become more difficult.

@baroni13 describe a system of this third kind in which composition is represented by the multiplication of words which are represented by variable order tensors. For example, a noun is a vector (1st order), while an adjective is a matrix (2nd order) because it is a function from nouns to nouns. A transitive verb is a 3rd order tensor because it first multiplies with an object, becoming a matrix, and then with a subject, becoming a vector. One important, and perhaps problematic, feature of this particular approach is that words with different syntactic categories have different shapes (i.e. they are different order tensors). In addition to the complications this approach creates for implementation theories, it requires that syntax be learned before, and independently from, semantics.


### A trivial merge operation
Although we see more potential for the second and third approaches to compositionality, they are far more difficult to pursue. Thus we present a merge function of the first type, which is designed to model an especially simple kind of compositionality that can be approximated fairly well with rules over categories (e.g. syntax). To construct this function, we begin with an example rule: $\n{NP} \rightarrow \n{D} \~ \n{N}$. Replacing explicit categories with similarity, and the non terminal \n{NP} with its compositional structure, we can say that, \n{[A B]} will be similar to \n{[D N]} if \n{A} is similar to \n{D} and \n{B} is similar to \n{N}. We are still left with the categories \n{D} and \n{N}. Thus, in line with the generalization algorithm discussed above, we replace a category label with a weighted average of all nodes. That is, upon creating the new node \n{[A B]}, we construct an initial row vector as the sum of every other chunk's row vector, weighted by the pairwise similarities of the respective constituents. Importantly, all parallel constituents must be similar (e.g. "the tasty macaroni" is not structurally similar to "ate tasty macaroni"). Thus we take the geometric mean (a multiplicative operation) of the pairwise similarities. The row for a newly constructed node $ab$ is

$$ \row_{ab} = \sum_{xy \in G} \row_{xy} \sqrt{\simil(a, b) \simil(x, y)}$$ {#eq:merge}




## Simulations
### Effect of dimensionality on storage capacity
To confirm that the sparse vector implementation of a graph reasonably matches a traditional graph representation, we compare the VectorGraph to a graph with true probabilities as edges, a ProbGraph. (Recall that VectorGraph edges roughly mirror probabilities). We expect that, as more edges are stored in a single vector, non-orthogonal index vectors will interfere with each other, resulting in noisy recovered edge weights. However, as the dimensionality of the vector increases, the chance of two random vectors being non-orthogonal decreases, making the edge weights more accurate.

To test this hypothesis, we provide a VectorGraph and a ProbGraph with the same random training data. If the VectorGraph implementation is sound, we expect the recovered edge weights after training to be very similar to the edge weights of the ProbGraph. However, because there is a chance that two randomly selected index vectors will not be orthogonal, the VectorGraph weights will be subject to some noise. We expect that the effect of noise will be greater for lower dimensionality vectors, and greater numbers of unique nodes. As shown in @fig:storage, the results match our expectations.

![Correlation of sparse vector and probabilistic graph edge weights with varying number of nodes and vector dimensionality. Node count increases from top to bottom. Dimensionality increases from left to right.](figs/compare_graphs.pdf){#fig:storage}

### Generalization
To test the generalization algorithm, we create a bigram model with a VectorGraph. The graph is trained on two corpora generated with probabilistic context free grammars. The grammars are nearly identical except for one determiner and one noun. The first has "that" and "table", while the second has "my" and "bunny". As a result, the strings "that bunny" and "my table" never occur in the combined corpus. However, the two determiners and the two nouns will have otherwise similar edge weights. If the generalization algorithm is successful, it will recognize this similarity and assign a non-zero weight the edge representing transitional probability between the unseen pairs. As shown in @fig:gen, both generalization algorithms are successful.

![Generalization. A non-zero edge weight is assigned to edges that were never bumped based on the pattern of determiner-noun connections.](figs/generalization.pdf){#fig:gen}

### Compositionality
To test the composition algorithm, we begin with the same bigram model as used in the previous simulation. We then create nodes representing noun phrases with all determiner-noun pairs, excluding \n{the} and \n{boy}. For each noun phrase, we assign high edge weights to \n{saw} and \n{ate}. As a test item, we create a new noun phrase node, \n{[the boy]}, either using the composition algorithm or not. Critically, \n{[the boy]} receives no direct training. We then measure edge weights from \n{[the boy]} to \n{saw} and \n{ate}. These will be near-zero when no composition is used. However, if the composition algorithm is successful, we expect \n{[the boy]} to have high edge weights to \n{saw} and \n{ate}. This is because previously encountered nodes composed of pairs of nodes like (\n{the}, \n{boy}) have high weights to \n{saw} and \n{ate}. As shown in @fig:composition, the results match our expectations.

![Composition. A newly created node "the boy" has edge-weights similar to an existing node "that table" because they are composed of similar elements. ](figs/composition.pdf){#fig:composition}

