from collections import OrderedDict
import sys

import numpy as np
import pandas as pd
from pandas import DataFrame
from joblib import Parallel, delayed

import corpora
import introspection
import utils
import sim_bleu
import sim_roc

from ngram import NGramModel
from numila import Numila

LOG = utils.get_logger(__name__, stream='INFO', file='WARNING')

class Dummy(object):
    """docstring for Dummy"""
    name = 'dummy'
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
        'prob': {'GRAPH': 'prob', 'EXEMPLAR_THRESHOLD': 0.05},
        'prob_ftp': {'GRAPH': 'prob', 'EXEMPLAR_THRESHOLD': 0.05, 'BTP_PREFERENCE': 0},
        'prob_btp': {'GRAPH': 'prob', 'EXEMPLAR_THRESHOLD': 0.05, 'BTP_PREFERENCE': 1000},
        'holo_flat': {'HIERARCHICAL': False},
        'prob_flat': {'GRAPH': 'prob', 'EXEMPLAR_THRESHOLD': 0.05, 'HIERARCHICAL': False},
        'holo_flat_full': {'PARSE': 'full', 'HIERARCHICAL': False},
        'prob_flat_full': {'GRAPH': 'prob', 'EXEMPLAR_THRESHOLD': 0.05, 'PARSE': 'full', 'HIERARCHICAL': False},
        'holo_bigram': {'EXEMPLAR_THRESHOLD': 2, 'BTP_PREFERENCE': 0},
        'prob_chunkless': {'GRAPH': 'prob', 'EXEMPLAR_THRESHOLD': 2},
        'prob_bigram': {'GRAPH': 'prob', 'EXEMPLAR_THRESHOLD': 2, 'BTP_PREFERENCE': 0},
        'dynamic1': {'DYNAMIC': 0.1},
        'dynamic3': {'DYNAMIC': 0.3},
        'dynamic5': {'DYNAMIC': 0.5},
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
        # Get model.
        if isinstance(name, tuple):
            func, args = name
            model = func(**args)
            name = model.name
        elif name in numila_params:
            model = Numila(name=name, **numila_params[name])
        elif name in other_models:
            model = other_models[name]()
        else:
            raise ValueError('Invalid model name {}'.format(name))

        # Add model to dictionary.
        if name in models:
            raise ValueError('two models have the same name')
        models[name] = model

    if parallel:
        jobs = (delayed(fit)(name, model, train_corpus) 
                for name, model in models.items())
        sys.setrecursionlimit(10000)  # for unpickling done in Parallel
        models = dict(Parallel(-2)(jobs))
    else:
        for name, m in models.items():
            m.fit(train_corpus)

    return models


def get_corpora(lang, kind, train_len, roc_len=100, bleu_len=100):
    if lang == 'toy2':
        import pcfg
        corpus = (s.split(' ') for s in pcfg.toy2())
    else:
        corpus = corpora.get_corpus(lang, kind)

    train_corpus = [next(corpus) for _ in range(train_len)]

    testable = (utt for utt in corpus if 2 < len(utt))
    roc_test_corpus = [next(testable) for _ in range(roc_len)]

    producable = (utt for utt in corpus if 2 < len(utt))
    bleu_test_corpus = [next(producable) for _ in range(bleu_len)]
    
    return {'train': train_corpus,
            'roc_test': roc_test_corpus,
            'bleu_test': bleu_test_corpus}



def run(models, lang, kind, train_len, roc_len=100, bleu_len=100):
    corpora = get_corpora(lang, kind, train_len, roc_len, bleu_len)
    models = get_models(models, corpora['train'])

    roc_df = DataFrame(list(sim_roc.main(models, corpora['roc_test'])))
    bleu_df = DataFrame(list(sim_bleu.main(models, corpora['bleu_test'])))
    chunk_df = DataFrame([{'model': name, **introspection.analyze_chunks(model)}
                         for name, model in models.items()])
    
    dfs = roc_df, bleu_df, chunk_df
    for df in dfs:
        df['lang'] = lang
        df['kind'] = kind

    return dfs


#########
def test(models, lang, train_len, roc_len=100, bleu_len=100):
    corpora = get_corpora(lang, train_len)
    models = get_models(models, corpora['train'])
    for utt in corpora['roc_test'][:5]:
        print('\n----------')
        print(*utt)
        for name, model in models.items():
            print(name)
            model.score(utt)

def model(train_len=1000, lang='english', kind='word', **params):
    # for testing
    model = Numila(**params)
    corpus = corpora.get_corpus(lang, kind)
    train_corpus = [next(corpus) for _ in range(train_len)]
    return model.fit(train_corpus)
#########

def main():
    models = [
        'random',
        'holo',
        'prob',
        'holo_flat',
        'holo_flat_full',
        'prob_bigram',
        'dynamic3'
    ]
    models = [
        'prob',
        'prob_flat',
        'prob_flat_full',
        'prob_chunkless',
        'prob_bigram',
        'prob_btp',
        'prob_ftp'
    ]
    langs = [
        #'toy2',
        'English',
        'Farsi',
        'German',
        'Hungarian',
        'Italian',
        'Japanese',
        'Spanish',
    ]

    kinds = ['word', 'syl', 'phone']
    train_len = 4000
    roc_len = 500
    bleu_len = 500
    
    jobs = [delayed(run)(models, lang, kind, train_len, roc_len, bleu_len)
            for lang in langs
            for kind in kinds]
    all_dfs = Parallel(n_jobs=-1)(jobs)


    for name, df in zip(['roc', 'bleu', 'chunk'], 
                        map(pd.concat, zip(*all_dfs))):
        file = 'pickles/prob_' + name
        df.to_pickle(file)
        print('wrote', file)    


if __name__ == '__main__':
    main()