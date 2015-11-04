import numpy as np
from pandas import DataFrame
from scipy import stats

from numila import Numila
import pcfg
import plotting
import utils
import vectors


#####################
## INSTROSPECTION  ##
#####################

def similarity_matrix(graph, round_to=None, num=None) -> DataFrame:
    """A distance matrix of all nodes in the graph."""
    if not graph.nodes:
        raise ValueError("Graph is empty, can't make distance matrix.")
    if num:
        #ind = np.argpartition(graph.counts, -num)[-num:]
        raise NotImplementedError()
    sem_vecs = [node.semantic_vec for node in graph.nodes]
    num_nodes = len(graph.nodes)
    matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            matrix[i,j] = matrix[j,i] = vectors.cosine(sem_vecs[i], sem_vecs[j])
    if round_to is not None:
        matrix = np.around(matrix, round_to)

    labels = graph.string_to_index.keys()
    return DataFrame(matrix, 
                     columns=labels,
                     index=labels)


def prediction_matrix(graph, kind='following', **distribution_args) -> DataFrame:
    df = DataFrame({node: node.distribution(kind, **distribution_args)
                    for node in graph.nodes})
    df.index = df.columns = graph.string_to_index.keys()
    return df


def describe(graph, node, n=5) -> str:
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


def word_sim(graph, word1, word2):
    return vectors.cosine(graph[word1].semantic_vec, graph[word2].semantic_vec)


def make_toy_corpus():
    with open('corpora/toy2.txt', 'w+') as f:
        for sent in pcfg.random_sentences('toy_pcfg2.txt', 1000):
            f.write(' '.join(sent))
            f.write('\n')


def plot_prediction(count_predictions, vec_predictions, word):
    the_dist = DataFrame({'count': count_predictions['the'],
                          'vector': vec_predictions['the']})
    the_dist.plot(kind='bar')
    import seaborn as sns
    sns.plt.show()


#################
## SIMULATIONS ##
#################

def cfg():
    for rate in [i/10 for i in range(1,11)]:
        print('\n=============\n'
              'RATE = {rate}'.format_map(locals()))
        graph = Numila(LEARNING_RATE=rate)
        with utils.Timer('train time'):
            with open('corpora/toy2.txt') as corpus:
                for i, s in enumerate(corpus.read().splitlines()):
                    if i % 100 == 99:
                        pass
                    graph.parse_utterance(s)  
                print('trained on {i} utterances'.format_map(locals()))

        sentences = ['Jack ate the hill',
                     'Jack ate the hill with my telescope',
                     'the boy under the table saw my cookie']
        for s in sentences:
            print(graph.parse_utterance(s))

        count_predictions = prediction_matrix(graph, use_vectors=False)
        for e in range(4,16,2):
            print('\ne = {e}'.format_map(locals()))
            for _ in range(3):
                print(graph.speak(exp=e))
            vec_predictions = prediction_matrix(graph, exp=e)
            print(stats.pearsonr(np.ravel(vec_predictions), np.ravel(count_predictions)))

        print('\nUSING COUNTS')
        for _ in range(9):
            print(graph.speak(use_vectors=False))



    #for e in range(6, 16, 2):
    #    plotting.heatmaps([prediction_matrix(graph, exp=e),
    #                       count_predictions,
    #                       ], titles=['Vector Prediction',
    #                                  'Count Prediction',
    #                                  ])
    
    
    #print('\nPREDICTIONS')
    #for w in ['saw', 'the', 'telescope', '[my hill]']:
    #    try:
    #        node = graph[w]
    #    except KeyError:
    #        print(w, 'not in graph')
    #    else:
    #        print('\n', describe(graph, node))

            #print(w, '->', node.predict())

    #print('\nPARSES')

    #for s in pcfg.random_sentences('toy_pcfg2.txt', 5):
    #    parse = str(graph.parse_utterance(s))
    #    print(parse)

    #print('\nSIMULATION')
    #s = 'the man saw Jack with the telescope'.split()
    #parse = str(graph.parse_utterance(s, verbose=True))


def syl():
    graph = Numila()
    for utt in utils.read_file('../PhillipsPearl_Corpora/English/test_sets/'
                               'English_syllable_test1.txt', r'/| '):
        graph.parse_utterance(utt)
    for utt in utils.read_file('../PhillipsPearl_Corpora/English/test_sets/'
                               'English_syllable_test2.txt', r'/| ')[:10]:
        print(graph.parse_utterance(utt))

    print(graph.chunkability(graph['vI'], graph['di']))
    print(graph.chunkability(graph['di'], graph['o']))
    print(graph.chunkability(graph['[vI di]'], graph['o']))

if __name__ == '__main__':
    cfg()
