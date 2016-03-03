import pandas as pd
import numpy as np
from scipy import stats
from joblib import Parallel, delayed

import utils
literal = utils.literal
from numila import Numila

LOG = utils.get_logger(__name__, stream='INFO', file='INFO')

def nu_grammaticality(model, utt):
    """Returns a grammaticality score for an utterance."""
    parse = model.parse(utt, learn=False)
    possible_chunks = len(parse.utterance) - 1
    
    chunk_ratio = parse.num_chunks / possible_chunks    
    
    if parse.chunkinesses:
        chunkedness = stats.gmean(parse.chunkinesses)
    else:  # no chunks made
        chunkedness = np.nan
    
    return (chunk_ratio, chunkedness)


def swapped(lst):
    """Yields all versions of a list with one adjacent pair swapped.

    [1, 2, 3] -> [2, 1, 3], [1, 3, 2]
    """
    for idx in range(len(lst) - 1):
        swapped = list(lst)
        swapped[idx], swapped[idx+1] = swapped[idx+1], swapped[idx]
        yield swapped

def create_test_corpus(corpus, num_utterances):
    """Creates a test corpus with grammatical and neighbor-swapped utterances."""
    usable = (utt for utt in corpus if len(utt) > 3)
    correct = utils.take_unique(usable, num_utterances)
    for original_idx, utt in enumerate(correct):
        yield (1, utt, original_idx)
        for foil in swapped(utt):
            # type, test utterance, original utterance
            yield(0, foil, original_idx)

def add_foils(corpus):
    """Creates a test corpus with grammatical and neighbor-swapped utterances."""
    for original_idx, utt in enumerate(corpus):
        yield (1, utt, original_idx)
        for foil in swapped(utt):
            # type, test utterance, original utterance
            yield(0, foil, original_idx)

def swapped_test(model, test_corpus):
    """Gets nu_grammaticality scores for items in the test_corpus"""
    for utt_type, utt, original in test_corpus:
        chunk_ratio, chunkedness = nu_grammaticality(model, utt)
        yield {'utterance_type': utt_type,
               'length': len(utt),
               'chunk_ratio': chunk_ratio,
               'chunkedness': chunkedness,
               'original': original}


def precision_recall(ranked):
    """Returns precisions and recalls on a test corpus with various thresholds.
    
    ranked is a list of utterance types, ranked by some metric.

    There is one data point for every correct utterance. These data points
    are the precision and recall if the model's grammaticality threshold is
    set to allow that utterance.
    """
    ranked = list(ranked)
    num_normal = ranked.count(1)

    # Iterate through utterances, starting with best-ranked. Every time
    # we find a normal one, we add a new data point: the recall and precision
    # if the model were to set the threshold to allow only the utterances seen
    # so far.
    normal_seen = 0
    num_seen = 0
    for utt_type in ranked:
        num_seen += 1
        if utt_type == 1:
            normal_seen += 1
            precision = normal_seen / num_seen
            recall = normal_seen / num_normal
            yield {'precision': precision,
                   'recall': recall,
                   'f_score': 2 * precision * recall / (precision + recall)}


def roc_curve(ranked):
    """Returns precisions and recalls on a test corpus with various thresholds.
    
    ranked is a list of utterance types, ranked by some metric.

    There is one data point for every utterance. These data points
    are the precision and recall if the model's grammaticality threshold is
    set to allow that utterance.
    """
    ranked = list(ranked)
    num_normal = ranked.count(1)

    correct_accepted = 0
    total_accepted = 0
    for utt_type in ranked:
        total_accepted += 1
        if utt_type == 1:
            correct_accepted += 1

        false_pos = (total_accepted - correct_accepted) / total_accepted
        true_pos = correct_accepted / num_normal
        yield {'false_pos': false_pos,
               'true_pos': true_pos}


def quick_test(train_len):
    corpus = utils.syl_corpus()
    test_corpus = create_test_corpus(corpus, num_utterances=100)
    train_corpus = [next(corpus) for _ in range(train_len)]
    model = Numila(**params).fit(train_corpus)

    df = pd.DataFrame(swapped_test(model, test_corpus))
    
    # We sort first by chunk_ratio, then by chunkedness. Thus this list
    # is sorted by nu_grammaticality()
    ranked = df.sort_values(['chunk_ratio', 'chunkedness'], ascending=False)
    ranked_types = list(ranked['utterance_type'])
    precision = pd.DataFrame(precision_recall(ranked_types))

    result = max(precision['f_score'])
    LOG.critical('best F: %s', result)
    return result


def eval_grammaticality_judgement(model, test_corpus):
    full_test_corpus = add_foils(test_corpus)
    ranked = list(sorted(full_test_corpus, key=lambda item: model.score(item[1]), reverse=True))
    ranked_types = (item[0] for item in ranked)
    scores = precision_recall(ranked_types)
    return max(scores, key=lambda item: item['f_score'])


def compare_models(models, test_corpus):
    for name, model in models.items():
        with utils.Timer(literal('{name} grammaticality')):
            score = eval_grammaticality_judgement(model, test_corpus)
            score['model'] = name
            yield score
