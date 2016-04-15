from collections import Counter
import numpy as np
import pandas as pd
from pandas import DataFrame

from numila import Numila
import utils
from ngram import NGramModel

LOG = utils.get_logger(__name__, stream='INFO', file='INFO')


def eval_production(model, test_corpus, metric_func):
    """Evaluates a model's performance on a test corpus based on a given metric.
    
    metric_func takes in two lists and returns a number between 0 and 1
    quantifying the similarity between the two lists.

    Returns a list of metric scores comparing an adult utterance to
    the model's reconstruction of that utterance from a
    scrambeled "bag of words" version of the utterance.
    """
    np.random.seed(0)
    for adult_utt in test_corpus:
        if len(adult_utt) < 2:
            continue  # can't evaluate a one word utterance
        words = list(adult_utt)[::-1]
        model_utt = model.speak(words)
        try:
            yield ({'length': len(model_utt), 
                'BLEU': metric_func(model_utt, adult_utt)})
        except:
            print('eval_production')
            import IPython; IPython.embed()

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

def quick_test(train_len=1000, **kwargs):
    corpus = utils.syl_corpus()
    test_corpus = [next(corpus) for _ in range(100)]
    train_corpus = [next(corpus) for _ in range(train_len)]
    model = Numila(**kwargs).fit(train_corpus)

    production_results = DataFrame(eval_production(model, test_corpus, common_neighbor_metric))
    result = production_results['BLEU'].mean()
    LOG.critical('quicktest: %s', result)
    return result

def compare_params(paramss, num_trials=5, train_len=1000):
    corpus = utils.syl_corpus()
    test_corpus = [next(corpus) for _ in range(100)]
    train_corpus = [next(corpus) for _ in range(train_len)]

    def data():
        for _ in range(num_trials):
            for params in paramss:
                model = Numila(**params).fit(train_corpus)
                df = DataFrame(eval_production(model, test_corpus, common_neighbor_metric))
                for k, v in params.items():
                    df[k] = v
                yield df

    df = pd.concat(data())
    df.to_pickle('pickles/comparison.')
    return df

def simple_test(model, test_corpus):
    results = DataFrame(eval_production(model, test_corpus, common_neighbor_metric))
    return results['BLEU'].mean()

def bleu_sim(models, test_corpus):
    for name, model in models.items():
        results = eval_production(model, test_corpus, common_neighbor_metric)
        for trial in results:
            trial['model'] = name
            yield trial


def old_bleu_sim(lang):
    models = lang['models']
    bleu = pd.DataFrame(compare_models(models, lang['bleu_test_corpus']))
    bleu['language'] = lang['language']
    return bleu


