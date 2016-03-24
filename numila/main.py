from collections import OrderedDict
import sys

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import metrics
from joblib import Parallel, delayed

import production
import introspection
import comprehension
import utils
from ngram import NGramModel
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
        'holo_flat': {'HIERARCHICAL': False},
        'prob_flat': {'GRAPH': 'probgraph', 'EXEMPLAR_THRESHOLD': 0.05,
                      'HIERARCHICAL': False},
        'holo_batch': {'PARSE': 'batch'},
        'prob_batch': {'GRAPH': 'probgraph', 'EXEMPLAR_THRESHOLD': 0.05, 
                       'PARSE': 'batch'},
        'prob_markov': {'GRAPH': 'probgraph', 'EXEMPLAR_THRESHOLD': 2},
        'prob_batch_markov': {'GRAPH': 'probgraph', 'EXEMPLAR_THRESHOLD': 2,
                            'PARSE': 'batch'},
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

    models = OrderedDict()
    for name in model_names:
        if not isinstance(name, str):
            model = name
            models[model.name] = model
        elif name in numila_params:
            models[name] = Numila(name=name, **numila_params[name])
        else:
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


def get_corpora(lang, train_len, roc_len=100, bleu_len=100):
    if lang == 'toy2':
        import pcfg
        corpus = (s.split(' ') for s in pcfg.toy2())
    else:
        corpus = utils.corpus(lang, 'syl')
    train_corpus = [next(corpus) for _ in range(train_len)]

    testable = (utt for utt in corpus if 2 < len(utt))
    #roc_test_corpus = utils.take_unique(testable, roc_len)
    roc_test_corpus = [next(testable) for _ in range(roc_len)]

    producable = (utt for utt in corpus if 2 < len(utt) < 7)
    #bleu_test_corpus = utils.take_unique(producable, bleu_len)
    bleu_test_corpus = [next(producable) for _ in range(bleu_len)]
    
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



def run(models, lang, train_len, roc_len=100, bleu_len=100):
    corpora = get_corpora(lang, train_len, roc_len, bleu_len)
    models = get_models(models, corpora['train'])

    #roc_test = [u for u in corpora['roc_test'] if len(u) == 5]  # FIXME
    roc_test = corpora['roc_test']

    roc_df = DataFrame(list(roc_sim(roc_test, models)))

    bleu_df = DataFrame(list(production.bleu_sim(models, corpora['bleu_test'])))

    chunk_df = DataFrame([{'model': name, **introspection.analyze_chunks(model)}
                         for name, model in models.items()])

    return roc_df, bleu_df, chunk_df


def test(models, lang, train_len, roc_len=100, bleu_len=100):
    corpora = get_corpora(lang, train_len)
    models = get_models(models, corpora['train'])
    for utt in corpora['roc_test'][:5]:
        print('\n----------')
        print(*utt)
        for name, model in models.items():
            print(name)
            model.score(utt)


def main():
    models = [
        'holo',
        'prob',
        'holo_flat',
        'prob_flat',
        'holo_batch',
        'prob_batch',
        'prob_markov',
        'prob_batch_markov',
    ]
    
    langs = [
        'English',
        'Spanish'
    ]
    
    all_dfs = []
    for lang in langs:
        #roc_df, bleu_df, chunk_df = run(models, lang, 4000, 500, 0)
        dfs = run(models, lang, 4000, 500, 0)
        for df in dfs:
            df['lang'] = lang
        all_dfs.append(dfs)

    for name, df in zip(['roc', 'bleu', 'chunk'], 
                        map(pd.concat, zip(*all_dfs))):
        df.to_pickle('pickles/' + name)
    


if __name__ == '__main__':
    main()
