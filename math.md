# Transitional probabilites and Barlow's principle
This is an attempted demonstration of how Barlow's principle of suspicious coincidence, as applied by the U-MILA model can be expressed in terms of forward and backward transitional probabilies.

## Definitions
- a *parse* is a traversal of the higraph that covers a given base token sequence. For example, [A [B C]] and [A B C] are two parses of "A B C".
- $count(A \Rightarrow B)$ is the number of times the bigram [A B] has occurred in all parses of all encountered sequences. This would be one entry in a collocation matrix.
- $count(A_1)$ is the number of times *A* has appeared before another node/token in a parse. It is equal to $\sum_{\lambda \in G} count(A \Rightarrow \lambda)$
- $count(B_2)$ is the number of times *B* has appeared after another node/token in a parse. It is equal to $\sum_{\lambda \in G} count(\lambda \Rightarrow B)$

Note that $count(A \Rightarrow B) = count(A_1 \cap B_2)$. The number of times A has been the initial element and B has been the second element in all bigrams in all parses.

## Probabilities
We can apply basic probability rules to derive the following. It's possible that the way I use $A_1$ here is not rigorous, but it makes intuitive sense to me.


$$P(A_1) = \frac{count(A_1)}{\sum_{\lambda \in G} count(\lambda_1)}$$

$$P(B_2) = \frac{count(B_2)}{\sum_{\lambda \in G} count(\lambda_2)}$$

$$P(A \Rightarrow B) = P(A_1 \cap B_2)$$

## Transitional Probabilities
Forward transitional probability (FTP) and backward transitional probability (BTP) are basically what they sound like.

$$FTP(A,B) = P(B_2|A_1) = \frac{P(B_2 \cap A_1)}{P(A_1)} $$

$$BTP(A,B) = P(A_1|B_2) = \frac{P(B_2 \cap A_1)}{P(B_2)} $$

By expanding the definitions of $P(A_1 \cap B_2)$ and $P(A_1)$, we can restate FTP:

$$FTP(A,B) = \frac{count(A \Rightarrow B)}{\sum_{\lambda \in G} count(A \Rightarrow \lambda)}$$

Similarly for BTP:

$$BTP(A,B) = \frac{count(A \Rightarrow B)}{\sum_{\theta \in G} count(\theta \Rightarrow B)}$$

Now we recall the definition of Barlow's principle:

$$Barlows(A, B) = \frac{count(A \Rightarrow B)}{\sum_{\lambda \in G} count(A \Rightarrow \lambda) \sum_{\theta \in G} count(\theta \Rightarrow B)}$$

Finally, multiplying FTP by BTP, we have:

$$FTP(A,B) BTP(A,B) = \frac{count(A \Rightarrow B)}{\sum_{\lambda \in G} count(A \Rightarrow \lambda)} \times \frac{count(A \Rightarrow B)}{\sum_{\theta \in G} count(\theta \Rightarrow B)}$$

It becomes clear here that Barlow's principle can be restated as:

$$Barlows(A+B) = \frac{FTP(A,B) BTP(A,B)}{count(A \Rightarrow B)}$$