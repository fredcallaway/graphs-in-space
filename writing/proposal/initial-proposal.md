

I plan to extend the language acquisition model described in @kolodny14. The work will most likely be entirely modeling (i.e. not experimental). We will use existing psycholinguistic data to test the performance of the model. After creating a satisfactory theoretical description, the model will be implemented in Python.

There are two specific extensions I propose:

1. Model the parsing and learning of language as one joint process that occurs incrementally, with cognitively interpretable states.
2. Model the learning of syntactic categories (i.e. parts of speech).

The original model, U-MILA, represents linguistic knowledge as a graph, with words and word-chunks as vertices and different relationships as edges. These edges can connect, for exapmle, two words that occur frequently together, or two words that occur in very similar contexts. Chunks are created when two words or existing chunks are found to co-occur very frequently. Chunks become vertices in the graph, allowing the model to learn about chunks in the same way it learns about words.

The proposed new model uses this same representational strategy, but adjusts the learning algorithm to be more psychologically realistic. The U-MILA algorithm involves searching far back in memory for possible chunks, something that humans are not likely able to do. I propose to replace this part of the algorithm with a modified Phrase Structure Grammar parsing algorithm that builds up a representaion of an utterance as it is heard.

The original model has strategies to represent similarities between words as well as a slot filler construction. Together, these two abilities are able to capture categorical knowledge to some degree, but generalization is limited. The proposed new model will attempt to connect the notions of word similarity and slot filling into one categorical representation.


