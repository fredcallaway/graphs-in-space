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
    return name, model.fit(train_corpus)

def main(train_len=5000, test_len=500):
    train_corpus, test_corpus = corpora(train_len, test_len)
    
    models = {
        'numila': Numila(),
        'no_chunk': Numila(EXEMPLAR_THRESHOLD=1),
        'probgraph': Numila(GRAPH='probgraph', EXEMPLAR_THRESHOLD=0.05),
        'dummy': Dummy(),
    }
    jobs = (delayed(fit)(name, model, train_corpus) 
            for name, model in models.items())
    sys.setrecursionlimit(10000)  # for unpickling done in Parallel
    models = dict(Parallel(-2)(jobs))
    models.update({'bigram': NGramModel(2).fit(train_corpus),  # can't do parallel
                   'trigram': NGramModel(3).fit(train_corpus),
                   'infant': Numila()})

    grammaticality = DataFrame(comprehension.compare_models(models, test_corpus))
    grammaticality.to_pickle('pickles/grammaticality.pkl')

    bleu = DataFrame(production.compare_models(models, test_corpus))
    bleu.to_pickle('pickles/production.pkl')


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


def gazorp(train_len=5000, test_len=500):
    train_corpus, test_corpus = corpora(train_len, test_len)

    for x in [0.5, 0.75, 1, 1.25, 1.5]:
        model = Numila(GRAPH='holograph', FTP_PREFERENCE=x).fit(train_corpus)
        print()
        print(production.simple_test(model, test_corpus))
        print(comprehension.eval_grammaticality_judgement(model, test_corpus))


def oldmain():
    comprehension.quick_test(train_len=5000, DECAY=0)
    comprehension.quick_test(GRAPH='probgraph', train_len=5000, DECAY=0)

    production.quick_test(GRAPH='probgraph',
                          CHUNK_THRESHOLD=0.2,
                          LEARNING_RATE=.1,
                          BIND=False,
                          train_len=5000)

    #quick_test(GRAPH='probgraph',
    #           CHUNK_THRESHOLD=0.2,
    #           LEARNING_RATE=.1,
    #           DECAY=0,
    #           BIND=False,
    #           train_len=5000)

    #for decay in (i * 2000 for i in range(1, 15)):
    #    print('decay:', decay)
    #    quick_test(GRAPH='probgraph',
    #               CHUNK_THRESHOLD=0.3,
    #               LEARNING_RATE=0.1,
    #               DECAY=decay,
    #               BIND=False,
    #               train_len=10000)




if __name__ == '__main__':
    try:
        optimizer()
    except:
        import traceback
        tb = traceback.format_exc()
        LOG.critical(tb)

