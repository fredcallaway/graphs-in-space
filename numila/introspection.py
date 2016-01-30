import pandas as pd
import numpy as np

import vectors
import utils

from collections import Counter

def analyze_chunks(model):
    chunks = [n for n in model.graph.nodes if hasattr(n, 'child1')]

    def size(chunk):
        return str(chunk).count('[') + 1

    return pd.DataFrame({'chunks': chunks,
                           'size': list(map(size, chunks))})




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
            model.parse_utterance(s)
    history = pd.Panel(history)
   
    return model, history


def main():
    from numila import Numila
    from production import eval_production, common_neighbor_metric

    for graph in ('probgraph', 'holograph'):
        corpus = utils.syl_corpus()
        test_corpus = [next(corpus) for _ in range(500)]
        train_corpus = [next(corpus) for _ in range(5000)]

        model = Numila(GRAPH=graph, EXEMPLAR_THRESHOLD=.1, LEARNING_RATE=1).fit(train_corpus)
        #model = Numila(GRAPH='probgraph', EXEMPLAR_THRESHOLD=1, LEARNING_RATE=1).fit(train_corpus)

        production_results = pd.DataFrame(eval_production(model, test_corpus, common_neighbor_metric))
        print(production_results['accuracy'].mean())

        df = analyze_chunks(model)
        print(df['size'].value_counts())
        #import IPython; IPython.embed()


if __name__ == '__main__':
    main()


