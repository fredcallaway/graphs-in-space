from collections import Counter
import numpy as np
import utils

LOG = utils.get_logger(__name__, stream='INFO', file='INFO')


def eval_production(model, test_corpus):
    """Evaluates a model's ability to order utterances in test_corpus."""
    np.random.seed(0)
    for adult_utt in test_corpus:
        if len(adult_utt) < 2:
            continue  # can't evaluate a one word utterance
        words = list(adult_utt)[::-1]
        model_utt = model.speak(words)
        yield {'length': len(model_utt), 
               'correct': int(model_utt == adult_utt),
               'BLEU2': bleu(model_utt, adult_utt, order=2),
               'BLEU3': bleu(model_utt, adult_utt, order=3),
               'BLEU4': bleu(model_utt, adult_utt, order=4),
               }

def bleu(lst1, lst2, order=2):
    """The percentage of N-grams (N = `order`) that are shared in two lists.

    Note that the metric is sensitive to the number of times a given
    pair occurs in each list. The lists are assumed to contain the
    same elements.
    
    [1,2,3] [3,1,2] -> 0.5
    [1,2,3,1,2], [1,2,2,3,1] -> 0.75
    """
    if order > len(lst1):
        return None
    ngrams1 = Counter(utils.neighbors(lst1, n=order))
    ngrams2 = Counter(utils.neighbors(lst2, n=order))
    num_shared = sum((ngrams1 & ngrams2).values())
    possible = sum(ngrams1.values())
    result = num_shared / possible
    assert 0 <= result <= 1
    return result

def main(models, test_corpus):
    for name, model in models.items():
        results = eval_production(model, test_corpus)
        for trial in results:
            trial['model'] = name
            yield trial


