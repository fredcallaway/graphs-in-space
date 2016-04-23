from sklearn import metrics

import utils

LOG = utils.get_logger(__name__, stream='INFO', file='INFO')


def add_foils(corpus):
    """Creates a test corpus with grammatical and neighbor-swapped utterances."""
    for original_idx, utt in enumerate(corpus):
        yield (1, utt, original_idx)
        for foil in swapped(utt):
            # type, test utterance, original utterance
            yield(0, foil, original_idx)

def swapped(lst):
    """Yields all versions of a list with one adjacent pair swapped.

    [1, 2, 3] -> [2, 1, 3], [1, 3, 2]
    """
    for idx in range(len(lst) - 1):
        swapped = list(lst)
        swapped[idx], swapped[idx+1] = swapped[idx+1], swapped[idx]
        yield swapped


def main(models, test_corpus):
    full_test = add_foils(test_corpus)
    y, targets, _ = zip(*full_test)

    for name, model in models.items():
        scores = model.map_score(targets)
        fpr, tpr, thresholds = metrics.roc_curve(y, scores)
        auc = metrics.auc(fpr, tpr)
        yield {'model': name,
               'fpr': fpr,
               'tpr': tpr,
               'auc': auc}