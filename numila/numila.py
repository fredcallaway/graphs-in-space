from collections import Counter, OrderedDict
import numpy as np
import pandas as pd
from scipy.spatial import distance
import time
from typing import Dict, List

from pcfg import random_sentences
from vectors import sparse_vector
from plotting import plot_mds, plot_dendrogram

VECTOR_SIZE = 1000
PERCENT_NON_ZERO = .01

FTP_PREFERENCE = .5

class Umila(object):
    """The premier language acquisition model"""
    def __init__(self):
        super(Umila, self).__init__()
        self.graph = Graph()
        self.last_node = self.graph['#']
        self.graph['#'].count += 1

    def read(self, token):
        """Reads one token and performs all associated computation"""
        self.graph.decay()
        node = self.graph[token]
        node.count += 1
        self.last_node.forward[node] += 1
        node.backward[self.last_node] += 1
        chunk = self.graph.chunk(self.last_node, node)
        if chunk:
            chunk.count += 1
            self.last_node = chunk
        else:
            self.last_node = node


    def read_file(self, file_path, split=' '):
        with open(file_path) as f:
            for line in f:
                start = time.time()
                if split:
                    tokens = line.split(split)
                else:
                    tokens = list(line)  # split by charcater
                del tokens[-1]  # remove trailing \n
                for t in tokens:
                    self.read(t)
                print('read %s tokens in %s seconds' % (len(tokens), time.time() - start))


class Graph(object):
    """docstring for Graph"""
    def __init__(self, size=10000):
        super(Graph, self,).__init__()
        # Each token gets an int ID which specifies its index
        # in self.nodes and self.activations.
        self.string_to_index = OrderedDict()  # type: Dict[str, int]
        self.nodes = []  # type: List[Node]
        self.activations = np.zeros(size)

    def __getitem__(self, token_string):
        if token_string in self.string_to_index:
            idx = self.string_to_index[token_string]
            return self.nodes[idx]
        else:  # new token
            idx = len(self.nodes)
            self.string_to_index[token_string] = idx
            self.nodes.append(Node(self, token_string, idx))
            self.activations[idx] = 1.0
            return self.nodes[idx]

    def __contains__(self, item):
        return item in self.string_to_index

    def decay(self):
        pass

    def chunk(self, node1, node2):
        chunk_string = '[{node1.string} {node2.string}]'.format(**locals())
        if '#' in chunk_string:
            # We assume that chunks cannot occur across utterance boundaries.
            return False
        if chunk_string in self:
            return self[chunk_string]
        elif min(node1.count, node2.count) < 3:
            return False
        else:
            ftp = distance.cosine(node1.semantic_vector, node2.after_context_vector)
            btp = distance.cosine(node1.before_context_vector, node2.semantic_vector)
            if (ftp + btp) / 2 < 0.5:
                return self[chunk_string]
            else:
                return False

    def distance_matrix(self, round_to=None):
        """A distance matrix of all nodes in the graph."""
        sem_vecs = [node.semantic_vector for node in self.nodes]
        num_nodes = len(self.nodes)
        matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i, num_nodes):
                matrix[i,j] = matrix[j,i] = distance.cosine(sem_vecs[i], sem_vecs[j])
        if round_to is not None:
            matrix = np.around(matrix, round_to)
        return pd.DataFrame(matrix, 
                            columns=self.string_to_index.keys(),
                            index=self.string_to_index.keys())

    def plot_mds(self):
        plot_mds(self.distance_matrix())

    def plot_dendrogram(self):
        plot_dendrogram(self.distance_matrix())


class Node(object):
    """A Node in a graph"""
    def __init__(self, graph, string, idx):
        self.string = string
        self.idx = idx
        self.count = 0
        self.forward = Counter()
        self.backward = Counter()
        self.id_vector = sparse_vector(VECTOR_SIZE, PERCENT_NON_ZERO)
        self.before_context_vector = np.roll(self.id_vector, -1)
        self.after_context_vector = np.roll(self.id_vector, 1)

    @property
    def semantic_vector(self):
        forward =  sum(node.after_context_vector * (weight / self.count) 
                       for node, weight in self.forward.items())
        backward =  sum(node.before_context_vector * (weight / self.count) 
                       for node, weight in self.backward.items())
        return FTP_PREFERENCE * forward + (1 - FTP_PREFERENCE) * backward

    def __hash__(self):
        return self.idx

    def __str__(self):
        followers = [(n[0].string, n[1]) for n in self.forward.most_common(5)]
        preceders = [(n[0].string, n[1]) for n in self.backward.most_common(5)]
        return ('Node({self.string})\n'
                'Followed by: {followers}\n'
                'Preceded by: {preceders}').format(**locals())


def similarity(node1, node2):
    return 1 - distance.cosine(node1.semantic_vector, node2.semantic_vector)


def main():
    m = Umila()
    for s in random_sentences('toy_pcfg2.txt', 1000):
        for t in s:
            m.read(t)
        m.read('#')  # sentence delimeter
    #print(m.graph['man'])
    #print(m.graph['telescope'])
    #print(m.graph['saw'])
    print(similarity(m.graph['cookie'], m.graph['telescope']))
    print(similarity(m.graph['cookie'], m.graph['saw']))
    print(m.graph.distance_matrix(round_to=1))
    m.graph.plot_dendrogram()
    import IPython; IPython.embed()




if __name__ == '__main__':
    print('\n')
    main()