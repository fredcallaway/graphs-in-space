
# Graphical models of language

The simplest graphical model of language is the bigram model, which treats language as a first order Markov process: each word is assumed to depend stochastically on only the previous word. A bigram model is generally represented as a transitional probability matrix, that is, a graph with words as nodes and transition probabilities as edges. In this model, an utterance can be produced by starting at a node $n_0$ (often a special \n{START} node), and then choosing the next a node $n_1$ with probability equal to the edge weight from $n_0$ to $n_1$. This process can be iterated until a termination criteria is reached (often the arrival at a special \n{STOP} node). Thus, generation is modeled as searching the graph for a path between the \n{STOP} and \n{STOP} nodes

Even under the false assumption that people speak purely based on statistical dependencies between words, the bigram model is fundamentally lacking. Language is rife with long distance dependencies such as "either-or" that a bigram model cannot possibly capture. One strategy to capture long distance dependencies is to increase the order of the Markov process. For example, a second order Markov process, or trigram model, assumes that a word depends on both the previous word and the word before that one. With some squinting, a trigram model can be represented as a standard directed graph with two words in each node. For example, the transitional probability $p(w_i = z | w_{i-1} = y, w_{i-2} = x)$ would be represented as the edge between the node $n_{xy}$ and $n_{yz}$.

However, increasing the Markov order has the undesirable side effect of exponentially increasing the dimensionality of the space. There are $n^N$ possible N-grams, where $N$ is the Markov order and $n$ is the vocabulary size. Thus, as $N$ increases, the percentage of grammatically valid N-grams that the learner will actually be exposed to will decrease exponentially. Many techniques in Natural Language Processing are designed to get around this problem of data sparsity, such as smoothing or variable order N-grams. For example, the back-off algorithm measures all N-gram probabilities of $N < N_{max}$, and dynamically decides which order, $N$ to use in a probability estimation based on the number of relevant N-grams it has stored for each $N$ [@katz87].

The ADIOS model [@solan05] explores an alternative technique for tracking long distance dependencies that aims to respect the hierarchical nature of language. Unlike N-gram models which always predict the single next word based on some number of previous words, ADIOS directly models the statistical dependencies between multi-word units, e.g. between "the dog" and "ate the steak". These multi-word units or "patterns" are constructed recursively through an iterative batch-learning algorithm. When two nodes (each of which may represent any number of words) are found to frequently occur adjacently in the corpus, they are combined into a new node. Later iterations may discover that this node occurs frequently with another node, allowing the creation of deep hierarchical patterns. The node composition function of ADIOS is a crucial development for graphical models of language. Nearly all modern syntactic theories take a binding function as a fundamental operation, although with different names: "Merge" [@chomsky95], "function application" [@steedman00], or "Unification" [@hagoort04]. The implementation of this function (by a modeler or a language learner) is a formidable task, as we discuss below.

The second major innovation of ADIOS is the use of different classes of nodes and edges to represent slot-filler constructions. When several patterns are found to mostly overlap, with one position containing different nodes in each pattern, an _equivalence class_ is identified. A unique class of node is used to represent the slot in the newly constructed pattern, with edges connecting to possible fillers. For example, upon finding the patterns "the big dog", "the nice dog", and "the recalcitrant dog", a new pattern would be created, "the \n{E1} dog", where \n{E1} is an equivalence class with edges pointing to "big", "nice", and "recalcitrant".

Although ADIOS demonstrated the utility of graphical representations in language modeling, the batch learning algorithm it employed casts some doubt on its relevance as a psychological model. However, this problem is not characteristic of graphical models in general. U-MILA [@kolodny15] is an incremental model based on ADIOS that more closely reflect human learning abilities. The model is incremental, passing through the corpus a single time, building up the graph from an initially clean slate. U-MILA was found to replicate a number of psycholinguistic results in word segmentation, category learning, and structural dependency learning.

Another recent model of language acquisition, the Chunk Based Learner [@mccauley14a] can also be expressed as a graph. This model is similar to the "bottom-up" mode of U-MILA in that chunks are created based on transitional probabilities between words and existing chunks. Slot filler constructions are not represented. The main psycholinguistic principle behind all these models is the idea of a "chunk", a sequence of words that is treated as a single unit. As in ADIOS and U-MILA, chunking is recursive: two chunks can be combined two form another chunk. Unlike ADIOS, however CBL and U-MILA do not maintain hierarchical order in the representation of multi-word sequences. Only the process of learning the chunks is hierarchical [c.f. @christiansen15 section 6.2]. Although CBL, U-MILA, and ADIOS have quite different theoretical motivations, they can all be expressed as a graph, facilitating direct comparison of the models.

