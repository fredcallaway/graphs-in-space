---
title:  Thesis progress report
author: Fred Callaway
date: \today
---

I track my progress on the project on GitHub^[https://github.com/fredcallaway/NU-MILA/]. This includes the model implementation, analyses of model behavior, and formal written descriptions of the model. Viewing this repository, and the history of developments is likely the best way to get a sense of my overall progress.


# Project description
Nümila is a model of language acquisition, comprehension, and production. The model aims to unite work in computational linguistics with work in psycholinguistics by combining hierarchically structured representations with a psychologically plausible learning model. It is almost universally agreed among linguists that language is structured hierarchically; however, there is a lack of evidence that such structures could be learned in a psychologically plausible way. As a result, many psycholinguists have turned to purely sequential models in an effort to avoid the implausible assumptions of most structured learning models. We suggest that there is a middle ground between these two approaches: By incorporating hierarchy into a chunk based learning model, we hope to provide evidence that (1) hierarchical structures can be learned in a psychologically feasible way, and (2) hierarchical structures allow a model to better capture linguistic structure.

A more complete description is available on the GitHub repository.

# Changes to the research plan
There has been one major change to the original research plan (reproduced below). I decided to delay step 1 (writing the introduction) until the beginning of the spring semester. The motivation for this is twofold. First, I have found that the process of modeling has changed how I view the model, and thus an introduction motivating the first iteration of the model would be innapropriate for the current iteration. It makes more sense to wait until the model reaches a largely stable state before writing an introduction. Second, I am writing a highly relevant paper for my Computational Psycholinguistics seminar, and I plan to use this material heavily in the introduction.

## Original research plan

1. Working from the Introduction section of the proposal, write the first section of the thesis, which motivates the work to be done.
2. Combine successful elements of previous computational models to create a new  computational model of the acquisition, comprehension, and production of language. Document major developments using the `git` version control system.
3. Analyze the basic computations employed by the model to determine its psychological and neural plausibility. Carefully document all findings.
4. Analyze the linguistic output of the model to determine its linguistic plausibility. Analyze the representations used by the model to determine ways in which the model does and does not match predictions from theoretical linguistics. Carefully document all findings.
5. `if (weeks_left > 8) then go to 2 else continue`
6. Compile the outlines generated in steps 2, 3, and 4 into prose sections describing (1) the structure and development of the model, (2) the psychological and neural plausibility of the model, and (3) the linguistic plausibility and significance of the model.
7. Write a final section which ties together the previous three sections, with special attention paid to the relevance of the model to psycholinguistics, theoretical linguistics, and cognitive science more generally.

# Progress
I have completed several iterations of steps 2 - 5. However, steps 3 and 4 have mostly been compressed to a single step: "Does the model behave at all reasonably?" At this point, the answer is a hesitant "yes", and we are now in a position to more carefully analyze the performance of the model.

The biggest development has been the use of high dimensional vectors to represent words, approximating the graphical structure of previous models. We suspect that this will make the model better able to generalize, as children are able to do. However, we are still unsure of exactly how to design the model to take advantage of the mathematical properties of high dimensional vectors to allow generalization. Shimon and I have been discussing this problem extensively, and I am optimistic about the prospect of implementing possible solutions over winter break.

Additionally, I have continued to read experimental psycholinguistic literature, keeping an eye out for phenomena that our model could likely capture. Conducting simulations of psycholinguistic experiments will be a major part of my work next semester, and I plan to begin a written list of potential simulations over winter break.

## Preliminary results
We have been testing the model on a corpus of child directed speech and an artifically generated corpus using a probabilistic context free grammar. In the spirit of open and reproducable research, these analyses are conducted in iPython notebooks that are publicly available on the GitHub repo.

At a high level, we see that the model is successfully capturing some regularities; however its generalization abilities are limited. We also see that the current system of hierarchical chunk generation does not contribute to performance. My intuition is that these two shortcomings are highly correlated. That is to say, I hope that improving the model's generalization algorithm will increase the utility of chunks.


