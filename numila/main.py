import numpy as np
import pandas as pd
from scipy import stats
import itertools
import joblib
from collections import Counter, defaultdict

from numila import Numila
from ngram import NGramModel
import plotting
import utils
import vectors

import re

from pprint import pprint, pformat

LOG = utils.get_logger(__name__, stream='INFO', file='INFO')

#####################
## INSTROSPECTION  ##
#####################


def similarity_matrix(model, round_to=None, num=None) -> pd.DataFrame:
    """A distance matrix of all nodes in the graph."""
    graph = model.graph
    if not graph.nodes:
        raise ValueError("Graph is empty, can't make distance matrix.")
    if num:
        #ind = np.argpartition(graph.counts, -num)[-num:]
        raise NotImplementedError()
    row_vecs = [node.row_vec for node in graph.nodes]
    num_nodes = len(graph.nodes)
    matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            matrix[i,j] = matrix[j,i] = vectors.cosine(row_vecs[i], row_vecs[j])
    if round_to is not None:
        matrix = np.around(matrix, round_to)

    labels = graph.string_to_index.keys()
    return pd.DataFrame(matrix, 
                     columns=labels,
                     index=labels)


def chunk_matrix(model, nodes='all') -> pd.DataFrame:
    if nodes is 'all':
        nodes = model.graph.nodes
    else:
        nodes = (model.graph.safe_get(node_str) for node_str in nodes)
        nodes = [n for n in nodes if n is not None]
    return pd.DataFrame({str(n1): {str(n2): model.chunkability(n1, n2) 
                                   for n2 in nodes} 
                        for n1 in nodes})


def word_sim(model, word1, word2):
    return vectors.cosine(model.graph[word1].row_vec, model.graph[word2].row_vec)


def plot_prediction(count_predictions, vec_predictions, word):
    the_dist = pd.DataFrame({'count': count_predictions['the'],
                          'vector': vec_predictions['the']})
    the_dist.plot(kind='bar')
    import seaborn as sns
    sns.plt.show()


def syntax_tree_link(parse):
    query = str(parse).replace('[', '[ ').replace(' ', '%20')
    return 'http://mshang.ca/syntree/?i=' + query


def node_frame(history, node, nodes):
    df = history.minor_xs(node)
    df['node'] = df.index
    mdf = pd.melt(df, id_vars=['node'], var_name='utterance', value_name='chunkability')
    return mdf[[x in nodes for x in mdf['node']]]



#################
## SIMULATIONS ##
#################

def exactly_equal_metric(lst1, lst2):
    """1 if the lists are the same, otherwise 0"""
    return 1 if lst1 == lst2 else 0

def common_neighbor_metric(lst1, lst2):
    """The percentage of adjacent pairs that are shared in two lists.
    Note that the metric is sensitive to the number of times a given
    pair occurs in each list.
    
    [1,2,3] [3,1,2] -> 0.5
    [1,2,3,1,2], [1,2,2,3,1] -> 0.75
    """
    pairs1 = Counter(utils.neighbors(lst1))
    pairs2 = Counter(utils.neighbors(lst2))
    num_shared = sum((pairs1 & pairs2).values())
    possible = sum(pairs1.values())
    result = num_shared / possible
    assert result >= 0 and result <= 1
    return result


def eval_production(production_func, test_corpus, metric_func):
    """Evaluates a model's performance on a test corpus based on a given metric.
    
    metric_func takes in two lists and returns a number between 0 and 1
    quantifying the similarity between the two lists.

    Returns a list of metric scores comparing an adult utterance to
    the model's reconstruction of that utterance from a
    scrambeled "bag of words" version of the utterance.
    """
    scores = defaultdict(list)
    for adult_utt in test_corpus:
        if len(adult_utt) < 2 or len(adult_utt) > 6:  # TODO!!
            continue  # can't evaluate a one word utterance
        words = np.copy(adult_utt)
        np.random.shuffle(words)
        model_utt = production_func(words)
        scores[len(model_utt)].append(metric_func(model_utt, adult_utt))
    return scores


def eval_comprehension(model, test_corpus):
    scores = defaultdict(list)
    for adult_utt in test_corpus:
        parse = model.parse_utterance(adult_utt)
        scores[len(parse.utterance)].append(parse.chunkiness)
    return scores


def eval_discrimination():
    corpus = syl_corpus()
    train = [next(corpus) for _ in range(5000)]
    numila = Numila().fit(train)
    
    test = (next(corpus) for _ in range(200))
    test = set(tuple(utt) for utt in test)  # no repeated utterances in test corpus
    foils = itertools.chain.from_iterable(swapped(utt) for utt in test)
    foils = set(tuple(utt) for utt in foils)

    with utils.Timer('chunkiness'):
        chunkiness = pd.DataFrame(
            [('true', numila.parse_utterance(utt).log_chunkiness)
             for utt in test] +
            [('foil', numila.parse_utterance(utt).log_chunkiness)
             for utt in foils]
            )
    with utils.Timer('precision'):
        precision = precision_recall(numila, test | foils, test)

        chunkiness.to_pickle('pickles/chunkiness.pkl')
        precision.to_pickle('pickles/precision.pkl')
        
    return chunkiness, precision

def swapped(utt):
    for idx in range(len(utt) - 1):
        swapped = list(utt)
        swapped[idx], swapped[idx+1] = swapped[idx+1], swapped[idx]
        yield swapped

def precision_recall(numila, test_corpus, correct):
    ranked = sorted(test_corpus, key=lambda u: numila_score_utterance(numila, u))

    data = []
    correct_seen = 0
    num_seen = 0
    for num_seen, utt in enumerate(ranked, start=1):
        if utt in correct:
            correct_seen += 1
            precision = correct_seen / num_seen
            recall = correct_seen / len(correct)
            data.append({'precision': precision, 'recall': recall})
    
    return pd.DataFrame(data)

def numila_score_utterance(numila, utt):
    """Lower is better."""
    parse = numila.parse_utterance(utt, learn=False)
    num_chunks = len(parse)
    neg_chunkiness = - parse.log_chunkiness
    return (num_chunks, neg_chunkiness)
            
#########

def recall_to_precision(recall, test_corpus, correct):
    """Returns the precision when the model achieves the given recall."""
    numila = Numila().fit(cfg_corpus())
    ranked = sorted(test_corpus, key=lambda u: numila_score_utterance(numila, u))
    
    num_required = np.ceil(len(correct) * recall)
    model_approved = list(keep_until_seen(ranked, correct, num_required=num_required))

    precision = num_required / len(model_approved)
    return precision
            
def keep_until_seen(iterable, targets, num_required=None):
    seen = 0
    if num_required == None:
        num_required = len(targets)
    for x in iterable:
        yield x
        if x in targets:
            seen += 1
        if seen >= num_required:
            return
    raise ValueError('Not enough targets in iterable.')

#########

def train(corpus, utterances=None, track=None, sample_rate=100, model_params={}):
    history = {}
    model = Numila(**model_params)
    with utils.Timer('train time', LOG.info):
        for i, s in enumerate(corpus):
            i += 1  # index starting at 1
            if track and i % sample_rate == 0:
                history[i] = (chunk_matrix(model, nodes=track))
            if i == utterances:
                break
            model.parse_utterance(s)
        LOG.info('trained on {} utterances'.format(i))
   
    history = pd.Panel(history)

    return model, history


def cfg_corpus(n=None):
    corpus = utils.read_corpus('corpora/toy2.txt', num_utterances=n)
    return corpus

def syl_corpus(n=None):
    corpus = utils.read_corpus('../PhillipsPearl_Corpora/English/English-syl.txt',
                               token_delim=r'/| ', num_utterances=n)
    return corpus


def master():
    corpus = syl_corpus()
    train_corpus = [next(corpus) for _ in range(20000)]
    test_corpus = [next(corpus) for _ in range(5000)]
    
    columns = ['exemplar_threshold', 'generalize', 'chunk_learning', 'length', 'score']
    production_data = []
    comprehension_data = []

    all_params = [
                  ('EXEMPLAR_THRESHOLD', [0.2, 0.3, 0.4]),
                  ('GENERALIZE', [0, 0.1, 0.3]),
                  ('CHUNK_LEARNING', [0, 0.2, 0.4]),
                  ]

    for params in utils.generate_args(all_params):
        model, _ = train(train_corpus, model_params=params)

        args = [params['EXEMPLAR_THRESHOLD'],
                params['GENERALIZE'],
                params['CHUNK_LEARNING']]

        production_func = lambda words: utils.flatten_parse(model.speak(words))
        production_results = eval_production(production_func, test_corpus, common_neighbor_metric)
        LOG.critical('PRODUCTION: %s', sum(sum(production_results.values(), [])))
        for length, scores in production_results.items():
            for score in scores:
                production_data.append(args + [length, score])

        comprehension_results = eval_comprehension(model, test_corpus)
        LOG.critical('COMPREHENSION: %s', sum(sum(comprehension_results.values(), [])))
        for length, scores in comprehension_results.items():
            for score in scores:
                comprehension_data.append(args + [length, score])

    pd.DataFrame(comprehension_data, columns=columns).to_pickle('pickles/comprehension.pkl')
    pd.DataFrame(production_data, columns=columns).to_pickle('pickles/production.pkl')


def quick_test(**kwargs):
    train_corpus = [next(syl_corpus()) for _ in range(2000)]
    test_corpus = [next(syl_corpus()) for _ in range(300)]
    model = Numila(**kwargs).fit(train_corpus)

    production_results = eval_production(model.speak, test_corpus, common_neighbor_metric)
    flat = sum(production_results.values(), [])
    result = sum(flat) / len(flat)
    LOG.critical('QUICK: %f3.3', result)
    return result


def main():
    production_data = []
    columns = ['train_len', 'speaker', 'length', 'score']
    for train_len in [1000, 2000, 4000, 8000]:
        corpus = syl_corpus()
        train_corpus = [next(corpus) for _ in range(train_len)]
        test_corpus = [next(corpus) for _ in range(1000)]

        speakers = {}


        bigram = NGramModel(2).fit(train_corpus)
        speakers['bigram'] = lambda words: bigram.speak(words)

        trigram = NGramModel(3).fit(train_corpus)
        speakers['trigram'] = lambda words: trigram.speak(words)

        numila = Numila().fit(train_corpus)
        speakers['numila'] = lambda words: utils.flatten_parse(numila.speak(words))

        no_chunk, _ = Numila(EXEMPLAR_THRESHOLD=1).fit(train_corpus)
        speakers['no_chunk'] = lambda words: utils.flatten_parse(no_chunk.speak(words))

        speakers['dummy'] = lambda words: words

        for speaker, func in speakers.items():
            production_results = eval_production(func, test_corpus, common_neighbor_metric)
            for length, scores in production_results.items():
                for score in scores:
                    production_data.append([train_len, speaker, length, score])

    pd.DataFrame(production_data, columns=columns).to_pickle('pickles/simple_production.pkl')


if __name__ == '__main__':
    eval_discrimination()


