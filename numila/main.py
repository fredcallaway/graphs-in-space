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


def corpora(train_len, test_len, corpus='syl'):
    if corpus == 'syl':
        corpus = utils.syl_corpus()
    elif corpus == 'pcfg':
        import pcfg
        corpus = (s.split(' ') for s in pcfg.toy2())
    usable = (utt for utt in corpus if 3 < len(utt) < 7)
    test = utils.take_unique(usable, test_len)
    train = [next(corpus) for _ in range(train_len)]
    LOG.debug('returned corpora')
    return train, test


def fit(name, model, train_corpus):
    return name, model.fit(train_corpus)


def get_models(train_corpus, parallel=False):
    models = {
        #'holo': Numila(),
        #'holo5k': Numila(DIM=5000),
        #'holo2k': Numila(DIM=2000),
        #'dynamic_01': Numila(DYNAMIC=0.1, name='dynamic_01'),
        #'dynamic_05': Numila(DYNAMIC=0.5, name='dynamic_05'),
        #'dynamic_10': Numila(DYNAMIC=1, name='dynamic_10'),
        'compose': Numila(COMPOSITION=True, DIM=5000),
        'compose2': Numila(COMP2=1, DIM=5000),
        #'convolve': Numila(COMPOSITION=True, BIND_OPERATION='convolution', DIM=5000),
        #'dynamic.1': Numila(DYNAMIC=0.1),
        #'dynamic.05': Numila(DYNAMIC=0.05),
        #'prob': Numila(GRAPH='probgraph', EXEMPLAR_THRESHOLD=0.05),
        'prob_chunkless': Numila(GRAPH='probgraph', EXEMPLAR_THRESHOLD=1,
                                 name='prob_chunkless'),
        #'trigram': NGramModel(3, '-addsmooth .0001'),
        #'random': Dummy()
    }
    if parallel:
        jobs = (delayed(fit)(name, model, train_corpus) 
                for name, model in models.items())
        sys.setrecursionlimit(10000)  # for unpickling done in Parallel
        models = dict(Parallel(-2)(jobs))
    else:
        for name, m in models.items():
            m.fit(train_corpus)

    return models

def main(train_len=3000, test_len=300, corpus='syl'):
    assert 0
    #train_corpus, test_corpus = corpora(train_len, test_len, corpus)
    #models = get_models(train_corpus)

    #for name, model in models.items():
    #    scores = model.map_score(None)
    #    auc = metrics.roc_auc_score(y, scores)
    #    print(name, auc)


def optimizer(train_len=5000, test_len=500):
    train_corpus, test_corpus = corpora(train_len, test_len)
    
    paramss = utils.generate_args([
        ('GRAPH', ['holograph']),
        ('MEMORY_SIZE', [2, 3, 4, 5, 6]),
    ]) + utils.generate_args([
        ('GRAPH', ['probgraph']),
        ('EXEMPLAR_THRESHOLD', [0.05]),
        ('MEMORY_SIZE', [2, 3, 4, 5, 6])
    ])
    def make_name(params):
        a = params['GRAPH'][0]
        b = params['MEMORY_SIZE']
        return a, b

    models = {make_name(ps): Numila(**ps) for ps in paramss}

    jobs = (delayed(fit)(name, model, train_corpus) 
            for name, model in models.items())
    sys.setrecursionlimit(10000)  # for unpickling done in Parallel
    models = dict(Parallel(-2)(jobs))

    grammaticality = DataFrame(comprehension.compare_models(models, test_corpus))
    grammaticality.to_pickle('pickles/opt-grammaticality.pkl')

    bleu = DataFrame(production.compare_models(models, test_corpus))
    bleu.to_pickle('pickles/opt-production.pkl')


def setup(lang, train_len=4000):
    # Create corpora.
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

    models = get_models(train_corpus)
    # models = get_ngrams(train_corpus)
    
    return {'language': lang,
            'roc_test_corpus': roc_test_corpus,
            'bleu_test_corpus': bleu_test_corpus,
            'models': models}

def roc(model, y, targets):
    scores = model.map_score(targets)
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    return fpr, tpr

def roc_lang(lang):
    full_test = comprehension.add_foils(lang['roc_test_corpus'])
    y, targets, _ = list(zip(*full_test))

    for name, model in lang['models'].items():
        fpr, tpr = roc(model, y, targets)
        auc = metrics.auc(fpr, tpr)
        yield {'language': lang['language'],
               'model': name,
               'fpr': fpr,
               'tpr': tpr,
               'ROC area under curve': auc}


if __name__ == '__main__':
    main(100, 100, 'pcfg')
    setup('toy2', 800)
