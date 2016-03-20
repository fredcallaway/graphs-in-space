import sys
from joblib import Parallel, delayed
import numpy as np
from pandas import DataFrame
from sklearn import metrics
import joblib

import production
import introspection
import comprehension
import utils
from ngram import NGramModel, get_ngrams
from numila import Numila

LOG = utils.get_logger(__name__, stream='INFO', file='WARNING')

class Dummy(object):
    """docstring for Dummy"""
    def fit(self, corpus):
        return self

    def speak(self, words):
        utt = list(words)
        np.random.shuffle(utt)
        return utt

    def score(self, utt):
        return np.random.random()

    def map_score(self, utts):
        return list(map(self.score, utts))


def fit(name, model, train_corpus):
    return name, model.fit(train_corpus)


def get_models(model_names, train_corpus, parallel=False):
    numila_params = {
        'holo': {},
        'prob': {'GRAPH': 'probgraph', 'EXEMPLAR_THRESHOLD': 0.05},
        'batch': {'PARSE': 'batch'},
        'prob_chunkless': {'GRAPH': 'probgraph', 'EXEMPLAR_THRESHOLD': 1}
    }

    other_models = {
        'trigram': lambda: NGramModel(3, '-addsmooth .0001'),
        'random': lambda: Dummy(),
    }
        #'holo5k': Numila(DIM=5000),
        #'holo2k': Numila(DIM=2000),
        #'dynamic_01': Numila(DYNAMIC=0.1, name='dynamic_01'),
        #'dynamic_05': Numila(DYNAMIC=0.5, name='dynamic_05'),
        #'dynamic_10': Numila(DYNAMIC=1, name='dynamic_10'),
        #'compose': Numila(COMPOSITION=True, DIM=5000),
        #'compose2': Numila(COMP2=1, DIM=5000),
        #'convolve': Numila(COMPOSITION=True, BIND_OPERATION='convolution', DIM=5000),
        #'dynamic.1': Numila(DYNAMIC=0.1),
        #'dynamic.05': Numila(DYNAMIC=0.05),

    models = {}
    for name in model_names:
        try:
            models[name] = Numila(name=name, **numila_params[name])
        except KeyError:
            models[name] = other_models[name]()

    if parallel:
        jobs = (delayed(fit)(name, model, train_corpus) 
                for name, model in models.items())
        sys.setrecursionlimit(10000)  # for unpickling done in Parallel
        models = dict(Parallel(-2)(jobs))
    else:
        for name, m in models.items():
            m.fit(train_corpus)

    return models


def get_corpora(lang, train_len):
    if lang == 'toy2':
        import pcfg
        corpus = (s.split(' ') for s in pcfg.toy2())
    else:
        corpus = utils.corpus(lang, 'syl')
    train_corpus = [next(corpus) for _ in range(train_len)]

    testable = (utt for utt in corpus if 2 < len(utt))
    #roc_test_corpus = utils.take_unique(testable, 100)
    roc_test_corpus = [next(testable) for _ in range(100)]

    producable = (utt for utt in corpus if 2 < len(utt) < 7)
    #bleu_test_corpus = utils.take_unique(producable, 100)
    bleu_test_corpus = [next(producable) for _ in range(100)]
    
    return {'train': train_corpus,
            'roc_test': roc_test_corpus,
            'bleu_test': bleu_test_corpus}

def roc_sim(test_corpus, models):
    full_test = comprehension.add_foils(test_corpus)
    y, targets, _ = list(zip(*full_test))

    for name, model in models.items():
        scores = model.map_score(targets)
        fpr, tpr, thresholds = metrics.roc_curve(y, scores)
        auc = metrics.auc(fpr, tpr)
        yield {'model': name,
               'fpr': fpr,
               'tpr': tpr,
               'auc': auc}

def run(lang, train_len, models):
    corpora = get_corpora(lang, train_len)
    models = get_models(models, corpora['train'])
    roc_results = roc_sim(corpora['roc_test'], models)
    roc_df = DataFrame(list(roc_results), columns=['model', 'auc'])
    return roc_df


def main():
    models = ['prob_chunkless', 'batch']
    df = run('English', 1000, models)
    print(df)

if __name__ == '__main__':
    main()
