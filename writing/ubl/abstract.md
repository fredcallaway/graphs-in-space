
What representations underlie language use? Identifying representations that are both psychologically plausible and also powerful enough to capture the complexity of language is a critical task for usage-based language researchers. @solan05 suggest a graph as one possible such representation, demonstrating that a graphical model can capture some of the regularity and structure of natural language. Graphs are appealing to usage-based researchers because they are simple, domain-general, and able to capture graded systematicity. However, it remains to be seen whether they could be implemented in the human brain. To begin to answer this question, we implement an approximate graph representation using a vector space model. We then apply this distributed graph representation in a novel online language learning model. Our results indicate that graphs can be approximately represented in a neurally plausible way, and also that such a representation can be useful in learning the structure of language.

- graph review
    - @solan05 basic idea of transitional probabilities, hierarchical nodes
    - @kolodny15 incremental, psycholinguistic modeling

- vectors review
    - @deerwester90 LSA, early use of vectors
    - @plate95 is this a good general introduction to vector models
    - @jones07 modern use of vectors, very similar to our approach
    - @sahlgren08 permutations instead of convolution (worth mentioning?)

- graph description
    - N x N adjacency matrix to N x D matrix
    - "id vector" and "row vector"
        - AKA "environment vector" and "semantic vector"
    - multiple edge types
    - bump_edge and edge_weight operations
- numila
    - highly incremental form of ADIOS
    - strict working memory and processing constraints
    - hopefully, a generalization scheme!