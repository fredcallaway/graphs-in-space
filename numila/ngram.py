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
    def __init__(self, order):
        self.order = order
        os.makedirs('_srilm/', exist_ok=True)
        self.dir = tempfile.mkdtemp(dir='_srilm/')

    def fit(self, train_data):
        with utils.Timer('Ngram train time'):
            with open(self.dir + '/train.txt', 'w+') as f:
                for line in train_data:
                    f.write(' '.join(line))

            cmd = ('ngram-count '
                   '-text {self.dir}/train.txt '
                   '-order {self.order} '
                   '-lm {self.dir}/model.lm '
                   '-kndiscount '
                   ).format_map(locals())

            subprocess.check_call(cmd, shell=True)
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
             '-debug 1 '
             '-ppl -'
             ).format_map(locals())

      out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
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

    def score(self, utterance):
        return next(self.perplexity([utterance]))

    def __del__(self):
        # This is not great practice for cleaning up, but it will have to do.
        shutil.rmtree(self.dir)




def main():
    import production
    import main
    from pandas import DataFrame
    train, test = main.corpora(5000, 100)
    model = NGramModel(2).fit(train)
    trials = DataFrame(production.eval_production(model, test, production.common_neighbor_metric))
    print(trials['accuracy'].mean())


if __name__ == '__main__':
    main()


