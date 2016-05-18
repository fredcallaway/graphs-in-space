"""A wrapper around SRILM"""

import itertools
import subprocess
import numpy as np
import tempfile
import utils
import re
import os
import shutil

LOG = utils.get_logger(__name__, stream='INFO', file='INFO')

class NGramModel(object):
    """A wrapper around SRILM"""
    def __init__(self, order, smoothing='', sri_args=''):
        self.order = order
        self.smoothing = smoothing
        self.sri_args = ' '.join(sri_args)
        os.makedirs('_srilm/', exist_ok=True)
        self.dir = tempfile.mkdtemp(dir='_srilm/')

    def fit(self, train_data):
        with utils.Timer('Ngram train time'):
            with open(self.dir + '/train.txt', 'w+') as f:
                for line in train_data:
                    f.write(' '.join(line))

            cmd = ('ngram-count '
                   '-text {dir}/train.txt '
                   '-order {order} '
                   '{smoothing} '
                   '{sri_args} '
                   '-lm {dir}/model.lm '
                   ).format_map(self.__dict__)

            subprocess.check_output(cmd, shell=True)
            os.remove(self.dir + '/train.txt')
        return self

    def logprob(self, ngram):
        assert 0
        ngram = ' '.join(ngram)
        cmd = ('echo "{ngram}" | '
               'ngram -order self.order '
               '-lm {self.dir}/model.lm '
               '-ppl -'
               ).format_map(locals())

        subprocess.check_output(cmd, shell=True)

    def perplexity(self, utterances):
        utts_string = '\n'.join(' '.join(u) for u in utterances)
        cmd = ('echo "{utts_string}" | '
               'ngram -order {self.order} '
               '-lm {self.dir}/model.lm '
               '{self.sri_args} '
               '-debug 1 '
               '-ppl -'
               ).format_map(locals())
        out = subprocess.check_output(cmd, shell=True)
        perplex_re = re.compile(rb'\d+ zeroprobs, logprob= \S+ ppl= ([\d.e+]+) ppl1= ')

        for block in out.split(b'\n\n')[:-1]:
            line = block.rsplit(b'\n', 1)[-1]
            yield float(perplex_re.match(line).group(1))

    def speak(self, words, verbose=False):
        """Returns a single node containing all of `words`."""
        assert len(words) < 7
        utts = list(itertools.permutations(words))
        perplexities = list(self.perplexity(utts))
        best_idx = np.argmin(perplexities)
        return utts[best_idx]

    def score(self, utt):
        return - next(self.perplexity([utt]))

    def map_score(self, utts):
        return [-p for p in self.perplexity(utts)]

    def __del__(self):
        # This is not great practice for cleaning up, but it will have to do.
        shutil.rmtree(self.dir)


def get_ngrams(train_corpus):
    models = {
        'add_bi': NGramModel(2, '-addsmooth .0001'),
        'add_tri': NGramModel(3, '-addsmooth .0001'),
        'gt_bi': NGramModel(2,),
        'gt_tri': NGramModel(3,),
        'wbdiscount': NGramModel(2, '-wbdiscount'),
        'ndiscount': NGramModel(2, '-ndiscount'),
        'kndiscount': NGramModel(2, '-kndiscount'),
        'unk_wbdiscount': NGramModel(2, '-wbdiscount', sri_args='-unk'),
        'unk_ndiscount': NGramModel(2, '-ndiscount', sri_args='-unk'),
        'unk_kndiscount': NGramModel(2, '-kndiscount', sri_args='-unk'),
        'unk_add': NGramModel(2, '-addsmooth .0001', sri_args='-unk'),
    }
    for m in models.values():
        m.fit(train_corpus)
    return models


def main():
    import main
    train_corpus, test_corpus = main.corpora(4000, 10)
    models = get_ngrams(train_corpus)
    test_corpus = main.joblib.load('pickles/test100')
    full_test = main.comprehension.add_foils(test_corpus)
    y, targets, _ = list(zip(*full_test))

    score_funcs = {name: model.map_score for name, model in models.items()}
    scores = {name: score(targets)
              for name, score in score_funcs.items()}

    print(scores['bigram'][:10])
    print(scores['kndiscount'][:10])
    print(scores['ndiscount'][:10])
    print(scores['addsmooth'][:10])

def test_score():
    from collections import defaultdict
    import main
    train_corpus, test_corpus = main.corpora(4000, 100)
    model = NGramModel(2).fit(train_corpus)
    test_corpus = list(test_corpus)
    scores = model.map_score(test_corpus)
    lens = map(len, test_corpus)
    
    len_scores = defaultdict(list)
    for ln, sc in zip(lens, scores):
        len_scores[ln].append(sc)

    avgs = {ln: np.mean(scores) for ln, scores in len_scores.items()}
    print(avgs)


if __name__ == '__main__':
    test_score()


