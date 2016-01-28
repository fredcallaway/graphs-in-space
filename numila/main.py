import numpy as np
import pandas as pd
import itertools
from collections import Counter, defaultdict

from numila import Numila
from ngram import NGramModel
import utils



LOG = utils.get_logger(__name__, stream='INFO', file='INFO')


class Dummy(object):
    """docstring for Dummy"""
    def fit(self, corpus):
        return self

    def speak(self, words):
        utt = list(words)
        np.random.shuffle(utt)
        return utt


## COMPARISONS ##

def compare_ngram():
    data = []
    for train_len in [1000, 2000, 4000]:
        corpus = syl_corpus()
        test_corpus = [next(corpus) for _ in range(500)]
        train_corpus = [next(corpus) for _ in range(train_len)]

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
