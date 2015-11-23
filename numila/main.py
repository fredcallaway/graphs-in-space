import numpy as np
import pandas as pd
from scipy import stats
import itertools
import joblib

from numila import Numila
import pcfg
import plotting
import utils
import vectors


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
        nodes = (model.graph.get(node_str, None) for node_str in nodes)
        nodes = [n for n in nodes if n is not None]
    return pd.DataFrame({str(n1): {str(n2): model.chunkability(n1, n2) 
                                   for n2 in nodes} 
                        for n1 in nodes})

def describe(graph, node, n=5) -> str:
    raise NotImplementedError()
    f_vec = node.distribution('following', use_vectors=True)
    p_vec = node.distribution('preceding', use_vectors=True)

    f_count = node.distribution('following', use_vectors=False)
    p_count = node.distribution('preceding', use_vectors=False)

    f_spear = stats.spearmanr(f_vec, f_count)
    p_spear = stats.spearmanr(p_vec, p_count)

    # only print the top n nodes
    def top_n(dist):
        top_n_indices = list(np.argpartition(dist, -n)[-n:])
        top_n_indices.sort(key=lambda i: dist[i], reverse=True)
        return [(graph.nodes[i], round(float(dist[i]), 3))
                for i in top_n_indices]

    f_vec = top_n(f_vec)
    p_vec = top_n(p_vec)
    f_count = top_n(f_count)
    p_count = top_n(p_count)

    return ('\n'.join([
            'Node({node.string})',
            'Followed by (count): {f_count}',
            'Followed by (vector): {f_vec}',
            '   Spearnman: {f_spear}',
            'Preceded by (count: {p_count}',
            'Preceded by (vector): {p_vec}',
            '   Spearman: {p_spear}',
            ]).format(**locals()))


def word_sim(model, word1, word2):
    return vectors.cosine(model.graph[word1].row_vec, model.graph[word2].row_vec)


def make_toy_corpus():
    with open('corpora/toy2.txt', 'w+') as f:
        for sent in pcfg.random_sentences('toy_pcfg2.txt', 1000):
            f.write(' '.join(sent))
            f.write('\n')


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

def slot_sim():
    model, history = train(cfg_corpus(), sample_rate=None)


def train(corpus, utterances=1000, track='all', sample_rate=100, model_params={}):
    history = {}
    graph = Numila(**model_params)
    with utils.Timer('train time'):
        for i, s in enumerate(corpus):
            i += 1  # index starting at 1
            if track and i % sample_rate == 0:
                history[i] = (chunk_matrix(graph, nodes=track))
            if i == utterances:
                break
            graph.parse_utterance(s)
        print('trained on {} utterances'.format(i))
   
    #joblib.dump(graph, 'pickles/cfg_model.pkl')
    history = pd.Panel(history)
    history.to_pickle('pickles/cfg_history.pkl')

    return graph, history


def cfg_corpus():
    corpus = utils.read_corpus('corpora/toy2.txt')
    return corpus

def syl_corpus():
    corpus = utils.read_corpus('../PhillipsPearl_Corpora/English/English-syl.txt',
                               token_delim=r'/| ')
    return corpus

if __name__ == '__main__':
    model, history = train(cfg_corpus())
