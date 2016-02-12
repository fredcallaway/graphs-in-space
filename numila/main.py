import sys

from joblib import Parallel, delayed
import numpy as np
from pandas import DataFrame

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


def corpora(train_len, test_len):
    corpus = utils.syl_corpus()
    usable = (utt for utt in corpus if 3 < len(utt) < 7)
    test = utils.take_unique(usable, test_len)
    train = [next(corpus) for _ in range(train_len)]
    return train, test


def fit(name, model, train_corpus):
    if name == 'generalized':
        return name, model.fit(train_corpus, lap=10)
    return name, model.fit(train_corpus)

import joblib

def get_models(train_corpus, parallel=False):
    models = {
        'generalized': Numila(GENERALIZE=('full', 0.3)),
        'numila': Numila(),
        'probgraph': Numila(GRAPH='probgraph', EXEMPLAR_THRESHOLD=0.05),
        'no_chunk': Numila(EXEMPLAR_THRESHOLD=1),
        'dummy': Dummy()
    }
    if parallel:
        jobs = (delayed(fit)(name, model, train_corpus) 
                for name, model in models.items())
        sys.setrecursionlimit(10000)  # for unpickling done in Parallel
        models = dict(Parallel(-2)(jobs))
    else:
        for name, m in models.items():
            if name == 'generalized':
                m.fit(train_corpus, lap=10)
            else:
                m.fit(train_corpus)

    models.update({'bigram': NGramModel(2).fit(train_corpus),  # can't do parallel
                   'trigram': NGramModel(3).fit(train_corpus)})

    return models

def main(train_len=3000, test_len=300):
    train_corpus, test_corpus = corpora(train_len, test_len)
    models = get_models(train_corpus)

    for name, m in models.items():
        joblib.dump(m, 'pickles/' + name)

    try:
        grammaticality = DataFrame(comprehension.compare_models(models, test_corpus))
        grammaticality.to_pickle('pickles/grammaticality.pkl')
        bleu = DataFrame(production.compare_models(models, test_corpus))
        bleu.to_pickle('pickles/production.pickles')
    except KeyboardInterrupt:
        pass

    import IPython; IPython.embed()



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
        return a, bs

    models = {make_name(ps): Numila(**ps) for ps in paramss}

    jobs = (delayed(fit)(name, model, train_corpus) 
            for name, model in models.items())
    sys.setrecursionlimit(10000)  # for unpickling done in Parallel
    models = dict(Parallel(-2)(jobs))

    grammaticality = DataFrame(comprehension.compare_models(models, test_corpus))
    grammaticality.to_pickle('pickles/opt-grammaticality.pkl')

    bleu = DataFrame(production.compare_models(models, test_corpus))
    bleu.to_pickle('pickles/opt-production.pkl')



def test_gen(train_len=2000, test_len=200):
    train_corpus, test_corpus = corpora(train_len, test_len)
    model = Numila(GENERALIZE=('full', 0.3)).fit(train_corpus, lap=10)

    import IPython; IPython.embed()

    print(production.simple_test(model, test_corpus))
    print(comprehension.eval_grammaticality_judgement(model, test_corpus))




if __name__ == '__main__':
    try:
        test_gen()
    except:
        import traceback
        tb = traceback.format_exc()
        LOG.critical(tb)

