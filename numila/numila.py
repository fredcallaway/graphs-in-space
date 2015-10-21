"""


"""


from collections import Counter, OrderedDict, defaultdict
import re
import numpy as np
import pandas as pd
from scipy.spatial import distance
import time
from typing import Dict, List
from typed import typechecked
from utils import get_logger

from pcfg import random_sentences
from vectors import sparse_vector
from plotting import plot_mds, plot_dendrogram

VECTOR_SIZE = 1000
PERCENT_NON_ZERO = .01

CHUNK_THRESHOLD = 0.4
MIN_NODE_COUNT = 3

FTP_PREFERENCE = .2
MEMORY_SIZE = 4

LOG = get_logger(__name__, stream='INFO', file='INFO')

class Numila(object):
    """The premier language acquisition model"""
    def __init__(self) -> None:
        super(Numila, self).__init__()
        self.graph = Graph()

    def parse_utterance(self, utterance):
         return Parse(self.graph, utterance)


class Node(object):
    """A Node in a graph.

    Attributes:
        string: e.g. [the [big dog]]
        idx: an int identifier
        forward_edges: the number of times each other node in the graph has been
                 after this node in the memory window
        backward_edges: the number of times each other node in the graph has been
                 before this node in the memory window
        id_vector: a random sparse vector
    """
    def __init__(self, graph, string, idx) -> None:
        self.string = string
        self.idx = idx
        self.count = 0

        self.forward_edges = Counter()
        self.backward_edges = Counter()

        self.id_vector = sparse_vector(VECTOR_SIZE, PERCENT_NON_ZERO)
        self.before_context_vector = np.roll(self.id_vector, -1)
        self.after_context_vector = np.roll(self.id_vector, 1)

    #@property
    #def count(self):
    #    return self.graph.counts[self.idx]

    @property
    @typechecked
    def semantic_vector(self) -> np.ndarray:
        if not (self.forward_edges or self.backward_edges):
            # No information about this node
            LOG.debug('Using 0 semantic_vector for {self}'.format_map(locals()))
            return np.ones(VECTOR_SIZE)

        ftp_vec = sum(node.after_context_vector * (weight / self.count) 
                          for node, weight in self.forward_edges.items())
        btp_vec = sum(node.before_context_vector * (weight / self.count) 
                           for node, weight in self.backward_edges.items())
        
        return (     FTP_PREFERENCE  * ftp_vec + 
                (1 - FTP_PREFERENCE) * btp_vec)

    def __hash__(self) -> int:
        return self.idx

    @property
    def description(self) -> str:
        followers = [(n[0].string, n[1]) for n in self.forward_edges.most_common(5)]
        preceders = [(n[0].string, n[1]) for n in self.backward_edges.most_common(5)]
        return ('Node({self.string})\n'
                'Followed by: {followers}\n'
                'Preceded by: {preceders}').format(**locals())

    def __repr__(self):
        return self.string

    def __str__(self):
        return self.string


class Graph(object):
    """docstring for Graph"""
    def __init__(self, size=100000) -> None:
        super(Graph, self,).__init__()
        # Each token gets an int ID which specifies its index
        # in self.nodes and self.activations.
        self.string_to_index = OrderedDict()  # type: Dict[str, int]
        self.nodes = []  # type: List[Node]
        self.activations = np.zeros(size)
        self.counts = np.zeros(size)

    def __getitem__(self, node_string) -> Node:
        if node_string in self.string_to_index:
            idx = self.string_to_index[node_string]
            return self.nodes[idx]
        else:
            if ' ' in node_string:
                raise ValueError('That chunk is not in the graph.')
            # node_string represents a new token, so we make a new Node for it
            return self.create_node(node_string)

    def __contains__(self, item) -> bool:
        return item in self.string_to_index

    def create_node(self, node_string) -> Node:
        idx = len(self.nodes)
        new_node = Node(self, node_string, idx)
        self.nodes.append(new_node)
        self.string_to_index[node_string] = idx
        self.activations[idx] = 1.0
        return new_node

    def decay(self) -> None:
        pass

    def chunk(self, node1, node2) -> Node:
        """Returns a chunk of node1 and node2 if the chunk is in the graph.

        If the chunk doesn't exist, we check if it should be created. It is
        returned if it is created."""
        chunk_string = '[{node1.string} {node2.string}]'.format_map(locals())
        if chunk_string in self:
            return self[chunk_string]
        else:
            # consider making a new node
            if self.chunkability(node1, node2) > CHUNK_THRESHOLD:
                return self.create_node(chunk_string)
            else:
                return None

    def chunkability(self, node1, node2) -> float:
        """How well two nodes form a chunk."""
        if node1.string is '#' or node2.string is '#':
            # We assume that chunks cannot occur across utterance boundaries.
            return 0.0
        ftp = cosine_similarity(node1.semantic_vector, node2.after_context_vector)
        btp = cosine_similarity(node1.before_context_vector, node2.semantic_vector)
        return (ftp + btp) / 2

    def distance_matrix(self, round_to=None, num=None) -> pd.DataFrame:
        """A distance matrix of all nodes in the graph."""
        if num:
            ind = np.argpartition(self.counts, -num)[-num:]
            raise NotImplementedError()
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

    def plot_mds(self) -> None:
        plot_mds(self.distance_matrix())

    def plot_dendrogram(self, **kwargs) -> None:
        plot_dendrogram(self.distance_matrix())

    def write_csv(self, features=None):
        """Creates distances.csv, a distance matrix of all MetaImages in self."""
        self.distance_matrix().to_csv('distances.csv')


class Parse(list):
    """docstring for ParseState"""
    def __init__(self, graph, utterance) -> None:
        super(Parse, self).__init__()
        self.graph = graph

        for token in utterance:
            self.shift(token)
            self.update_temporal_weights(MEMORY_SIZE)
            if len(self) >= 4:  # fill memory before trying to chunk
                self.make_best_chunk(MEMORY_SIZE)

        # Process the tail end. We have to shrink memory_size to prevent
        # accessing elements that fell out of the 4 item memory window.
        for memory_size in range(MEMORY_SIZE-1, 1, -1):
            self.update_temporal_weights(memory_size)
            self.make_best_chunk(memory_size)

    def shift(self, token) -> None:
        LOG.debug('shift {token}'.format_map(locals()))
        node = self.graph[token]
        node.count += 1
        self.append(node)

    def update_temporal_weights(self, memory_size) -> None:
        memory_size = min(memory_size, len(self))
        for i in range(1, memory_size):
            node = self[-i]
            previous_node = self[-i-1]
            previous_node.forward_edges[node] += 1
            node.backward_edges[previous_node] += 1

    def make_best_chunk(self, memory_size) -> None:
        memory_size = min(memory_size, len(self))
        LOG.debug('memory_size = {memory_size}'.format_map(locals()))
        if memory_size < 2:
            return  # can't make a chunk with less than 2 elements
        chunkabilities = []
        for i in range(1, memory_size):
            node = self[-i]
            previous_node = self[-i-1]
            chunkability = self.graph.chunkability(previous_node, node)
            LOG.debug('chunkability({previous_node}, {node}) = {chunkability}'.format_map(locals()))
            chunkabilities.append(chunkability)
    
        best = chunkabilities.index(max(chunkabilities))
        chunk = self.graph.chunk(self[-best-2], self[-best-1])
        if chunk:
            # combine the two nodes into one chunk
            chunk.count += 1
            self[-best-2] = chunk
            del self[-best-1]

    def __str__(self):
        return super(Parse, self).__str__().replace(',', '')


def cosine_similarity(v1, v2) -> float:
    return 1.0 - distance.cosine(v1, v2)

def node_similarity(node1, node2) -> float:
    return cosine_similarity(node1.semantic_vector, node2.semantic_vector)

def draw_tree(tree_string):
    raise NotImplementedError()

    from nltk import Tree
    from nltk.draw.util import CanvasFrame
    from nltk.draw import TreeWidget

    cf = CanvasFrame()
    tree = Tree.fromstring(tree_string.replace('[','(').replace(']',')') )
    cf.add_widget(TreeWidget(cf.canvas(), tree), 10, 10)
    cf.print_to_file('tree.ps')
    cf.destroy

def read_file(file_path, token_delim=' ', utt_delim='\n') -> List[List[str]]:
    utterances = []
    with open(file_path) as f:
        for utterance in re.split(utt_delim, f.read()):
            if token_delim:
                tokens = re.split(token_delim, utterance)
            else:
                tokens = list(utterance)  # split by character
            utterances.append(tokens)

    print(len(utterances), ...)
    return utterances

def cfg():
    m = Numila()

    for s in random_sentences('toy_pcfg2.txt', 1000):
        m.parse_utterance(s)

    for s in random_sentences('toy_pcfg2.txt', 100):
        parse = str(m.parse_utterance(s))
        print(parse)

    print(m.graph.chunkability(m.graph['the'], m.graph['telescope']))
    print(m.graph.chunkability(m.graph['telescope'], m.graph['saw']))

def syl():
    m = Numila()
    for utt in read_file('../PhillipsPearl_Corpora/English/test_sets/English_syllable_test1.txt', r'/| '):
        m.parse_utterance(utt)
    for utt in read_file('../PhillipsPearl_Corpora/English/test_sets/English_syllable_test2.txt', r'/| ')[:100]:
        print(m.parse_utterance(utt))

    print(m.graph.chunkability(m.graph['vI'], m.graph['di']))
    print(m.graph.chunkability(m.graph['di'], m.graph['o']))
    print(m.graph.chunkability(m.graph['[vI di]'], m.graph['o']))


def main() -> None:
    cfg()
    #syl()



if __name__ == '__main__':
    #print('\n')
    main()