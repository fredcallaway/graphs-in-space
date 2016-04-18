import pandas as pd
import numpy as np
from numila import Numila

import vectors
import utils
import pcfg
import main

def analyze_chunks(model):
    if model.name == 'dummy':
        return {'num_chunks': None,
                'average_size': None,
                'max_size': None}

    chunks = [n for n in model.graph.nodes if n.children]
    sizes = list(map(size, chunks))
    return {'num_chunks': len(chunks),
            'average_size': len(sizes) and np.mean(sizes),
            'max_size': max(sizes, default=0)}

def size(node):
    if not node.children:
        return 1
    else:
        return sum(map(size, node.children))




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


def track_training(model, corpus, utterances=None, track=None, sample_rate=100):
    history = {}
    with utils.Timer('train time'):
        for i, s in enumerate(corpus, start=1):
            if track and i % sample_rate == 0:
                history[i] = (chunk_matrix(model, nodes=track))
            if i == utterances:
                break
            model.parse(s)
    history = pd.Panel(history)
   
    return model, history



def main_():
    num_nodes = 10
    from holograph import HoloGraph
    graph = HoloGraph()
    ids = [str(i) for i in range(num_nodes)]
    for id in ids:
        node = graph.create_node(id)
        graph.add(node)
    



if __name__ == '__main__':
    main_()