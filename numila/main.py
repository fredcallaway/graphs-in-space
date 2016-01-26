import numpy as np
import pandas as pd
import itertools
from collections import Counter, defaultdict

from numila import Numila
from ngram import NGramModel
import utils
make = utils.make



LOG = utils.get_logger(__name__, stream='INFO', file='INFO')


class Dummy(object):
    """docstring for Dummy"""
    def fit(self, corpus):
        return self

    def speak(self, words):
        utt = list(words)
        np.random.shuffle(utt)
        return utt


#################
## SIMULATIONS ##
#################


## PRODUCTION ##

@make(pd.DataFrame)
def eval_production(model, test_corpus, metric_func):
    """Evaluates a model's performance on a test corpus based on a given metric.
    
    metric_func takes in two lists and returns a number between 0 and 1
    quantifying the similarity between the two lists.

    Returns a list of metric scores comparing an adult utterance to
    the model's reconstruction of that utterance from a
    scrambeled "bag of words" version of the utterance.
    """
    for adult_utt in test_corpus:
        if len(adult_utt) < 2 or len(adult_utt) > 6:  # TODO!!
            continue  # can't evaluate a one word utterance
        words = list(adult_utt)
        np.random.shuffle(words)
        model_utt = model.speak(words)
        yield ({'length': len(model_utt), 
                'accuracy': metric_func(model_utt, adult_utt)})

def exactly_equal_metric(lst1, lst2):
    """1 if the lists are the same, otherwise 0"""
    return 1 if lst1 == lst2 else 0

def common_neighbor_metric(lst1, lst2):
    """The percentage of adjacent pairs that are shared in two lists.
    Note that the metric is sensitive to the number of times a given
    pair occurs in each list.
    
    [1,2,3] [3,1,2] -> 0.5
    [1,2,3,1,2], [1,2,2,3,1] -> 0.75
    """
    pairs1 = Counter(utils.neighbors(lst1))
    pairs2 = Counter(utils.neighbors(lst2))
    num_shared = sum((pairs1 & pairs2).values())
    possible = sum(pairs1.values())
    result = num_shared / possible
    assert result >= 0 and result <= 1
    return result


## COMPREHENSION ##

def eval_discrimination(numila, test_corpus):
    full_test = create_test_corpus(test_corpus)
    grammaticality = score_utterances(numila, full_test)
    precision = precision_recall(grammaticality)

    #grammaticality.to_pickle('pickles/grammaticality.pkl')
    #precision.to_pickle('pickles/precision.pkl')
        
    return grammaticality, precision

def create_test_corpus(corpus):
    """Returns a test corpus with grammatical and neighbor-swapped utterances."""
    correct = set(tuple(utt) for utt in corpus if len(utt) > 1)
    foils = set(tuple(foil) for utt in correct for foil in swapped(utt))
    return [('normal', utt) for utt in correct] + [('swapped', utt) for utt in foils]

def swapped(lst):
    """Yields all versions of a list with adjacent pairs swapped.

    [1, 2, 3] -> [2, 1, 3], [1, 3, 2]
    """
    for idx in range(len(lst) - 1):
        swapped = list(lst)
        swapped[idx], swapped[idx+1] = swapped[idx+1], swapped[idx]
        yield swapped

def nu_grammaticality(numila, utt):
    """Returns a grammaticality score for an utterance. Lower is better. """
    parse = numila.parse_utterance(utt, learn=False)
    num_chunks = len(parse)
    chunk_surprisal = - parse.log_chunkiness
    return (num_chunks, chunk_surprisal)

@make(pd.DataFrame)
def score_utterances(numila, test_corpus):
    """Returns grammaticality scores for a test corpus."""
    for grammatical, utt in test_corpus:
        num_chunks, chunk_surprisal = nu_grammaticality(numila, utt)
        yield {'grammatical': grammatical,
               'length': len(utt),
               'num_chunks': num_chunks,
               'chunk_surprisal': chunk_surprisal}

@make(pd.DataFrame)
def precision_recall(grammaticality):
    """Returns precisions and recalls on a test corpus with various thresholds.

    There is one data point for every correct utterance. These data points
    are the precision and recall if the model's grammaticality threshold is
    set to allow that utterance.
    """
    # We sort first by num_chunks, then by chunk_surprisal, thus this list
    # is sorted by nu_grammaticality()
    ranked = grammaticality.sort_values(['num_chunks', 'chunk_surprisal'])
    num_correct = grammaticality['grammatical'].value_counts()[True]

    # Iterate through utterances, starting with best-ranked. Every time
    # we find a correct one, we add a new data point: the recall and precision
    # if the model were to set the threshold to allow only the utterances seen
    # so far.
    correct_seen = 0
    num_seen = 0
    for _, utt in ranked.iterrows():
        num_seen += 1
        if utt['grammatical']:
            correct_seen += 1
            precision = correct_seen / num_seen
            recall = correct_seen / num_correct
            yield {'precision': precision,
                   'recall': recall,
                   'F_score': np.sqrt(precision * recall)}

## COMPARISONS ##

def compare_ngram():
    data = []
    for train_len in [1000, 2000, 4000, 8000]:
        corpus = syl_corpus()
        train_corpus = [next(corpus) for _ in range(train_len)]
        test_corpus = [next(corpus) for _ in range(500)]

        models = {}
        models['bigram'] = NGramModel(2).fit(train_corpus)
        models['trigram'] = NGramModel(3).fit(train_corpus)
        models['numila'] = Numila().fit(train_corpus)
        models['no_chunk'], _ = Numila(EXEMPLAR_THRESHOLD=1).fit(train_corpus)
        models['dummy'] = Dummy()

        for name, model in models.items():
            results = eval_production(model, test_corpus, common_neighbor_metric)
            results['train length'] = train_len
            results['model'] = name
            data.append(results)

    df = pd.concat(data)
    df.to_pickle('pickles/ngram_comparison.pkl')
    return df


def compare_params():
    corpus = syl_corpus()
    train_corpus = [next(corpus) for _ in range(20000)]
    test_corpus = [next(corpus) for _ in range(5000)]
    
    all_params = [
                  ('EXEMPLAR_THRESHOLD', [0.2, 0.3, 0.4]),
                  ('GENERALIZE', [0, 0.1, 0.3]),
                  ('CHUNK_LEARNING', [0, 0.2, 0.4]),
                  ]

    production_data = []
    comprehension_data = []
    for params in utils.generate_args(all_params):
        numila = Numila(**params).fit(train_corpus)

        production_results = eval_production(numila, test_corpus, common_neighbor_metric)
        LOG.critical('PRODUCTION: %s', sum(production_results['accuracy']))
        for k, v in params.items():
            production_results[k] = v
        production_data.append(production_results)

        _, comprehension_results = eval_discrimination(numila, test_corpus)
        LOG.critical('COMPREHENSION: %s', max(comprehension_results['F_score']))
        for k, v in params.items():
            comprehension_results[k] = v
        comprehension_data.append(comprehension_results)

    pd.concat(comprehension_data).to_pickle('pickles/comprehension.pkl')
    pd.concat(production_data).to_pickle('pickles/production.pkl')


def quick_test(**kwargs):
    corpus = syl_corpus()
    train_corpus = [next(corpus) for _ in range(2000)]
    test_corpus = [next(corpus) for _ in range(300)]
    model = Numila(**kwargs).fit(train_corpus)

    production_results = eval_production(model.speak, test_corpus, common_neighbor_metric)
    result = production_results['accuracy'].mean()
    LOG.critical('QUICK: %f3.3', result)
    return result


if __name__ == '__main__':
    train_corpus = [next(syl_corpus()) for _ in range(2000)]
    test_corpus = [next(syl_corpus()) for _ in range(300)]
    numila = Numila().fit(train_corpus)
    eval_discrimination(numila, test_corpus)
