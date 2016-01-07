import numpy as np
import pandas as pd
import itertools
from collections import Counter, defaultdict

from numila import Numila
from ngram import NGramModel
import utils



LOG = utils.get_logger(__name__, stream='INFO', file='INFO')



#################
## SIMULATIONS ##
#################

def cfg_corpus(n=None):
    corpus = utils.read_corpus('corpora/toy2.txt', num_utterances=n)
    return corpus

def syl_corpus(n=None):
    corpus = utils.read_corpus('../PhillipsPearl_Corpora/English/English-syl.txt',
                               token_delim=r'/| ', num_utterances=n)
    return corpus


## PRODUCTION ##
def eval_production(production_func, test_corpus, metric_func):
    """Evaluates a model's performance on a test corpus based on a given metric.
    
    metric_func takes in two lists and returns a number between 0 and 1
    quantifying the similarity between the two lists.

    Returns a list of metric scores comparing an adult utterance to
    the model's reconstruction of that utterance from a
    scrambeled "bag of words" version of the utterance.
    """
    scores = defaultdict(list)
    for adult_utt in test_corpus:
        if len(adult_utt) < 2 or len(adult_utt) > 6:  # TODO!!
            continue  # can't evaluate a one word utterance
        words = np.copy(adult_utt)
        np.random.shuffle(words)
        model_utt = production_func(words)
        scores[len(model_utt)].append(metric_func(model_utt, adult_utt))
    return scores

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
def eval_discrimination(num_train=5000, num_test=200):
    corpus = syl_corpus()
    train = [next(corpus) for _ in range(num_train)]
    numila = Numila().fit(train)
    test_corpus = create_test_corpus(corpus, num_test)

    grammaticality = score_utterances(numila, test_corpus)
    precision = precision_recall(grammaticality)

    #grammaticality.to_pickle('pickles/grammaticality.pkl')
    #precision.to_pickle('pickles/precision.pkl')
        
    return grammaticality, precision

def create_test_corpus(corpus, length):
    """Returns a test corpus with grammatical and neighbor-swapped utterances."""
    correct = utils.take_unique(corpus, length, filter=lambda utt: len(utt) > 1)
    foils = (foil for utt in correct for foil in swapped(utt))
    foils = set(tuple(utt) for utt in foils)  # ensure uniqueness
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

def score_utterances(numila, test_corpus):
    """Returns grammaticality scores for a test corpus."""
    data = []
    for grammatical, utt in test_corpus:
        num_chunks, chunk_surprisal = nu_grammaticality(numila, utt)
        data.append({'grammatical': grammatical,
                     'length': len(utt),
                     'num_chunks': num_chunks,
                     'chunk_surprisal': chunk_surprisal})
    return pd.DataFrame(data)

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
    data = []
    correct_seen = 0
    num_seen = 0
    for _, utt in ranked.iterrows():
        num_seen += 1
        if utt['grammatical']:
            correct_seen += 1
            precision = correct_seen / num_seen
            recall = correct_seen / num_correct
            data.append({'precision': precision, 'recall': recall})
    return pd.DataFrame(data)


###########

def master():
    corpus = syl_corpus()
    train_corpus = [next(corpus) for _ in range(20000)]
    test_corpus = [next(corpus) for _ in range(5000)]
    
    columns = ['exemplar_threshold', 'generalize', 'chunk_learning', 'length', 'score']
    production_data = []
    comprehension_data = []

    all_params = [
                  ('EXEMPLAR_THRESHOLD', [0.2, 0.3, 0.4]),
                  ('GENERALIZE', [0, 0.1, 0.3]),
                  ('CHUNK_LEARNING', [0, 0.2, 0.4]),
                  ]

    for params in utils.generate_args(all_params):
        model, _ = train(train_corpus, model_params=params)

        args = [params['EXEMPLAR_THRESHOLD'],
                params['GENERALIZE'],
                params['CHUNK_LEARNING']]

        production_func = lambda words: utils.flatten_parse(model.speak(words))
        production_results = eval_production(production_func, test_corpus, common_neighbor_metric)
        LOG.critical('PRODUCTION: %s', sum(sum(production_results.values(), [])))
        for length, scores in production_results.items():
            for score in scores:
                production_data.append(args + [length, score])

        comprehension_results = eval_comprehension(model, test_corpus)
        LOG.critical('COMPREHENSION: %s', sum(sum(comprehension_results.values(), [])))
        for length, scores in comprehension_results.items():
            for score in scores:
                comprehension_data.append(args + [length, score])

    pd.DataFrame(comprehension_data, columns=columns).to_pickle('pickles/comprehension.pkl')
    pd.DataFrame(production_data, columns=columns).to_pickle('pickles/production.pkl')


def quick_test(**kwargs):
    train_corpus = [next(syl_corpus()) for _ in range(2000)]
    test_corpus = [next(syl_corpus()) for _ in range(300)]
    model = Numila(**kwargs).fit(train_corpus)

    production_results = eval_production(model.speak, test_corpus, common_neighbor_metric)
    flat = sum(production_results.values(), [])
    result = sum(flat) / len(flat)
    LOG.critical('QUICK: %f3.3', result)
    return result


def main():
    production_data = []
    columns = ['train_len', 'speaker', 'length', 'score']
    for train_len in [1000, 2000, 4000, 8000]:
        corpus = syl_corpus()
        train_corpus = [next(corpus) for _ in range(train_len)]
        test_corpus = [next(corpus) for _ in range(1000)]

        speakers = {}


        bigram = NGramModel(2).fit(train_corpus)
        speakers['bigram'] = lambda words: bigram.speak(words)

        trigram = NGramModel(3).fit(train_corpus)
        speakers['trigram'] = lambda words: trigram.speak(words)

        numila = Numila().fit(train_corpus)
        speakers['numila'] = lambda words: utils.flatten_parse(numila.speak(words))

        no_chunk, _ = Numila(EXEMPLAR_THRESHOLD=1).fit(train_corpus)
        speakers['no_chunk'] = lambda words: utils.flatten_parse(no_chunk.speak(words))

        speakers['dummy'] = lambda words: words

        for speaker, func in speakers.items():
            production_results = eval_production(func, test_corpus, common_neighbor_metric)
            for length, scores in production_results.items():
                for score in scores:
                    production_data.append([train_len, speaker, length, score])

    pd.DataFrame(production_data, columns=columns).to_pickle('pickles/simple_production.pkl')


if __name__ == '__main__':
    eval_discrimination(10, 10)


