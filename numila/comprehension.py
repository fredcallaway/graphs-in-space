import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns

import utils
from numila import Numila



def nu_grammaticality(numila, utt):
    """Returns a grammaticality score for an utterance."""
    parse = numila.parse_utterance(utt, learn=False)
    possible_chunks = len(parse.utterance) - 1
    
    chunk_ratio = parse.num_chunks / possible_chunks    
    
    if parse.chunkinesses:
        chunkedness = stats.gmean(parse.chunkinesses)
    else:  # no chunks made
        chunkedness = np.nan
    
    return (chunk_ratio, chunkedness)


def swapped(lst):
    """Yields all versions of a list with one adjacent pair swapped.

    [1, 2, 3] -> [2, 1, 3], [1, 3, 2]
    """
    for idx in range(len(lst) - 1):
        swapped = list(lst)
        swapped[idx], swapped[idx+1] = swapped[idx+1], swapped[idx]
        yield swapped

def create_test_corpus(corpus, num_utterances):
    """Returns a test corpus with grammatical and neighbor-swapped utterances."""
    usable = (utt for utt in corpus if len(utt) > 3)
    correct = utils.take_unique(usable, num_utterances)
    foils = (foil for utt in correct for foil in swapped(utt))
    foils = set(tuple(utt) for utt in foils)  # ensure uniqueness
    return ([('normal', utt) for utt in correct] + 
            [('swapped', utt) for utt in foils])


def swapped_test(numila, test_corpus):
    for grammatical, utt in test_corpus:
        chunk_ratio, chunkedness = nu_grammaticality(numila, utt)
        yield {'utterance_type': grammatical,
               'length': len(utt),
               'chunk_ratio': chunk_ratio,
               'chunkedness': chunkedness}


def precision_recall(ranked):
    """Returns precisions and recalls on a test corpus with various thresholds.
    
    ranked is a list of utterance types, ranked by some metric.

    There is one data point for every correct utterance. These data points
    are the precision and recall if the model's grammaticality threshold is
    set to allow that utterance.
    """
    num_normal = ranked.count('normal')

    # Iterate through utterances, starting with best-ranked. Every time
    # we find a normal one, we add a new data point: the recall and precision
    # if the model were to set the threshold to allow only the utterances seen
    # so far.
    normal_seen = 0
    num_seen = 0
    for utt_type in ranked:
        num_seen += 1
        if utt_type == 'normal':
            normal_seen += 1
            precision = normal_seen / num_seen
            recall = normal_seen / num_normal
            yield {'precision': precision,
                   'recall': recall,
                   'F_score': np.sqrt(precision * recall)}


def main():
    corpus = utils.syl_corpus()
    train_corpus = [next(corpus) for _ in range(5000)]
    test_corpus = create_test_corpus(corpus, num_utterances=100)
    numila = Numila().fit(train_corpus)

    df = pd.DataFrame(swapped_test(numila, test_corpus))
    df.to_csv('swapped_test.csv')
    df.to_pickle('swapped.pkl')
    
    # We sort first by chunk_ratio, then by chunkedness. Thus this list
    # is sorted by nu_grammaticality()
    nu_ranked = df.sort_values(['chunk_ratio', 'chunkedness'], ascending=False)
    nu_ranked_types = list(nu_ranked['utterance_type'])
    nu_precision = precision_recall(nu_ranked_types)
    sns.jointplot('recall', 'precision', data=nu_precision)


if __name__ == '__main__':
    main()


