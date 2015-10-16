---
title:  Thesis proposal
author: Fred Callaway
abstract: |
    Computational models have had considerable success in modeling the acquisition, comprehension, and production of language.
    However, almost all models either rely on cognitively unrealistic computations or fail to capture the complexity of true natural language.
    This may be because most modeling efforts take into account either the constraints imposed by the linguistic data or the constraints imposed by cognition--rarely both.
    In the proposed project, we will attempt to carefully consider both of these constraints while constructing a computational model using suitable component of previous models.
    Hopefully, this will result in a model that more fully and accurately captures human language.
---

# Introduction
Comparing language to the communication systems of other animals, it is clear that language is quite unique. Only a human can communicate her frustration with an inconsiderate coworker by exhaling, vibrating her vocal cords, and moving her mouth and tongue in a particular way. However, despite this clear high-level distinction, there is little consensus on what _exactly_ makes language unique [e.g. @hockett68]. Similarly, it is clear that non-human primates are unable to learn human language, despite several attempts [e.g. @patterson78]. However, there is even less agreement on what uniquely human cognitive abilities underly this difference in performance.

Which cognitive abilities one identifies as critical for language depends largely on which external features of language are deemed most important. Authors who focus on the social aspects of language point to humans' unique social reasoning abilities [@worden98; @malle02]. Other authors who focus on the use of symbols in language, point to our referential abilities [@deacon98; @bates79]. Theoretical linguists, who focus mostly on the structure of language, claim it is the ability to _combine_ symbols that truly differentiates humans from other animals [@hauser02; @chomsky10].

The social and referential aspects of language, and the connection between these two aspects [@tomasello03] make for a fascinating and active field of research that we do not have space to discuss here. We will instead follow the linguists in focusing on the structure of language. Specifically, we wish to address two highly related questions:

1. What kinds of structure exist in language?
2. What cognitive abilities are necessary to learn and use those structures?

Each question is subject to its own empirical constraints: A satisfactory answer to question 1 must be able to account for the full range of utterances used in all human languages. <!-- TODO: good? -->
It should also explain how these structures determine the compositional meaning of a sentence. A satisfactory answer to question 2, on the other hand, must offer only computations that humans might realistically employ. What is "realistic" depends on empirical results in neuroscience and psychology. Besides these empirical constraints, the answers to these two questions constrain each other: Any structures proposed in answer 1 must be learnable and usable by the abilities proposed in answer 2, and the abilities proposed in answer 2 must be sufficient to learn and use the structures proposed in answer 1.


Linguistic data <- representational structures <-> cognitive abilities -> neuroscience & psychology

Figure: Theoretical dependencies in psycholinguistics


Most theoretical work puts more emphasis on just one question and its constraints. Transformational syntactic theories, for example, focus entirely on explaining the linguistic data, creating increasingly complex structures to explain them <!-- [citation] -->. They explicitly cast aside the constraint of cognitive abilities as mere issues of "performance" [@chomsky65]. As a result, these theories posit unwieldy representations that seem impossible to learn [@jackendoff11]. On the other end, SRN models restrict themselves to neurally inspired representations <!-- [although see citation] -->, but are only able to capture specific structures, often using toy corpora [@christiansen01; @french14]. This gap between models that take linguistic complexity seriously and models that take cognitive limitations seriously has resulted in a dearth of comprehensive psycholinguistic models.

This thesis will attempt to make a small step towards filling that gap. By carefully considering both the constraints imposed by linguistic data as well as those imposed by neuroscience and psychology, we have the best chance of understanding how language works.

# Previous models
Luckily for me, the gap described above is not entirely empty. Work done in part by Dr. Edelman has already made major progress in this direction. Two models in particular are especially good examples of models which conform to the constraints imposed by linguistic data and neuroscience. These are ADIOS [@solan05] and U-MILA [@kolodny14], which itself was based on ADIOS. These two models share three desirable attributes which set them apart from most models of language [but see @bod09, others?].

- They are _unsupervised_, able to capture linguistic structure based solely on examples of valid utterances. [^supervision]
- They are _generative_ in that they can produce and assign a probability to unseen sentences.
- They can operate on full size _natural language_ corpora, not just small artificial grammars

[^supervision]: The comeplete lack of supervision is actually a flaw in U-MILA. Semantic and social factors clearly provide important information to the learner [@frank09]. However, requiring no supervision is preferable to requiring unrealistic supervision like pre-parsed trees [e.g. @johnson02].

Lack of direct supervision is critical to maintain psychological plausibility. Any model that requires parsed trees as input data [e.g. @johnson02] cannot possibly describe the child's learning process. Meanwhile, generativity and scalability are essential attributes of a model that claims to capture the complexity of natural language. <!-- TODO: expand -->

A critical realization that gives these two models their power is that language knowledge can be represented as a graph. Graphs are an excellent formalism for studying language because they are both neurally plausible and highly expressive, especially well suited to represent the complex relationships between many different entities. For example, a word could commonly precede a phrase, it could be part of the phrase, or it could often be replaced by the phrase. These three types of relationships are fundamental in language, and they can all be represented at once in a graph.

## ADIOS
Of the three models discussed here, ADIOS is the most powerful, but least psychologically plausible. Like most symbolic language acquisition models [?], ADIOS is a _batch learning algorithm_, which means it operates on all the training data at once. It begins by creating a graph with one vertex for every word in the corpus, as well as special _begin_ and _end_ vertices. For each sentence in the corpus, it constructs a path from _begin_ to _end_. Each edge on the path is indexed by the order it appeared in the corpus, thus the initial graph is _isomorphic_ to the corpus.

The algorithm proceeds by searching the graph for significant _patterns_, which can be intuitively understood as word sequences that occur frequently in the corpus. ADIOS does this by following a random path from _begin_ to _end_ and looking at how many edges connect each word. Initially, this is equivalent to the number of sentences in which each pair of words occurred. By comparing the number of edges coming into a word, with the number of edges going from this word to another word, the model can calculate the _transitional probability_ between the two words. For example, say there are 4 edges coming into "hammer" and three edges going from "hammer" to "time". This means that "hammer" occurred 4 times in the corpus, and 3 of these times, "hammer" was followed by "time". We can thus calculate the probability of hearing "time" given that you just heard "hammer" as $P(time|hammer) = 0.75$.

If the model only preformed this one step, it would be equivalent to a bigram model. In reality, ADIOS detects patterns recursively, so the effective Markov order can be considerably higher than two.
<!-- TODO: fix following --> and it computes probabilities in both forward and backward directions. Without going into the details, ADIOS can recognize that "bread and butter" occurs frequently, and that "dog" is very often preceded by "the". The use of backward transitional probabilities, (i.e.
the probability of one word preceding another) is especially important because it allows the identification of patterns like "the dog", where "the" can be followed by many words, but "dog" can only be preceded by a few.

Once ADIOS has identified a significant pattern, it creates a new vertex representing the entire sequence of words. For every path that passes through the full sequence, ADIOS replaces the edge leading to the first word with one leading to the newly constructed vertex. Similarly, the edge out of the final word is replaced with one out of the new vertex. Thus, the original paths have been shortened, with common material compressed into a single unit.

By constructing these patterns, ADIOS represents the compositional structure of language. A critical aspect of this process is that pattern identification can occur recursively on existing patterns, allowing ADIOS to capure the hierarchical structure of language. For example, after identifying "hammer time" as pattern, ADIOS can later recognize "stop hammer time" as a pattern composed of "stop" and "hammer time". A pattern can thus be represented as a tree, or equivalently with brackets: [stop [hammer time]].

Any linguist reading this proposal will notice something important missing from the tree above, part of speech labels. ADIOS models this kind of categorical knowledge with _equivalence classes_. The intuitive observation is that similar words occur in similar contexts. If you ignore order and treat the context as a bag of words, you get _distributional semantic models_, such as Latent Semantic Analysis [@lund96]. By incorporating order, however, ADIOS is able to capture both syntactic and semantic patterns, although, ADIOS does not distinguish between the two. Syntactic patterns come out in the form of dependencies between increasingly abstract word classes.

<!-- 
In slightly more detail, the categorical generalization algorithm consists of searching the graph for paths that overlap for all but one vertex. Each of these vertices has thus been found to occur in one identical context. A new vertex is created ...
__[I don't understand this part very well yet. Maybe leave out this paragraph?]__
 -->


## U-MILA
U-MILA is an attempt to incorporate the main representational and statistical strategies of ADIOS into an incremental learning algorithm. This is a major step forward in modeling language acquisition, where the child clearly does not wait until age three to begin processing every utterance she has ever heard (and somehow memorized verbatim). This allows U-MILA to model a continuous trajectory of language acquisition rather than simply the end product or punctuated stages [as in @bod09]. Perhaps more importantly, U-MILA imposes a biologically feasible memory constraint by limiting the model's analysis to the last 50-300 words it has seen.

U-MILA has two independent learning mechanisms, termed "top-down segmentation" and "bottom-up chunking". Each of these mechanisms can be loosely viewed as a weakened version of the ADIOS algorithm. This results in a model which is much more psychologically realistic, while maintaining much of ADIOS's power.

U-MILA's top-down segmentation is similar to the full ADIOS algorithm, but applied to only the 50-300 most recently seen words. Like in ADIOS, the algorithm learns by searching for recurring word sequences. However, rather than examining all utterances in the corpus at once, it only searches recent memory. In addition to increasing psychological plausibility, this restriction may actually improve the ability of the model to identify meaningful chunks [@goldstein10]. If a partial overlap is found, but with some internal word varying, an equivalence class is created for these words. This is very similar in spirit to the method used by ADIOS.

U-MILA's bottom-up segmentation is similar to the ADIOS algorithm in that it uses information from all previously encountered utterances; however it only uses the simplest kind of information, bigram transitional probabilities. This requires only that the model remember how many times each possible pair of words has occurred. Keeping in mind that exact counts are a proxy for a fuzzier connection weight, this is biologically feasible. Consider the similarity to long term synaptic potentiation: "words that fire together, wire together". As in ADIOS, both forward and backward transitional probabilities are tracked; new vertices are created when a combination of these two probabilities exceeds a threshold. Unlike in ADIOS, however, new _chunks_ are created through concatenation rather than composition. This means that there is no distinction between [the [big dog]] and [[the big] dog]. Whether this is a desirable feature of a model is an important question that will be explored in the project.

The graphical structure of U-MILA makes production straightforward to model. To produce an utterance, the model simply follows an arbitrary path from the _begin_ node to the _end_ node. Any slots encountered are filled with possible fillers. Additionally, the model may randomly switch to a similar node at any time, a way to model categorical generalization. Similarly, the model can assign a probability to a possible utterance by summing the probabilities of all possible graph traversals that result in the utterance string. Using these tools, Kolodny et al. (2014) were able to test the model for its precision and recall, as well as on a variety of psycholinguistic phenomenon. The details will not be covered here, but the model outperformed a trigram model and performed well on an auxiliary fronting test. 

# The plan

1. Working from the Introduction section of this document, write the first section of the thesis, which motivates the work to be done.
2. Combine successful elements of previous computational models to create a new  computational model of the acquisition, comprehension, and production of language. Document major developments using the `git` version control system.
3. Analyze the basic computations employed by the model to determine its psychological and neural plausibility. Carefully document all findings.
4. Analyze the linguistic output of the model to determine its linguistic plausibility. Analyze the representations used by the model to determine ways in which the model does and does not match predictions from theoretical linguistics. Carefully document all findings.
5. `if (weeks_left > 8) then go to 2 else continue`
6. Compile the outlines generated in steps 2, 3, and 4 into prose sections describing (1) the structure and development of the model, (2) the psychological and neural plausibility of the model, and (3) the linguistic plausibility and significance of the model.
7. Write a final section which ties together the previous three sections, with special attention paid to the relevance of the model to psycholinguistics, theoretical linguistics, and cognitive science more generally.

# References
