from collections import OrderedDict
import itertools
from typing import Dict, List
import numpy as np
import utils
import vectors


LOG = utils.get_logger(__name__, stream='INFO', file='WARNING')

COUNT = 0

class Node(object):
    """A Node in a graph.

    Attributes:
        string: e.g. [the [big dog]]
        idx: an int identifier
        id_vec: a random sparse vector that never changes
    """
    def __init__(self, graph, id_string, id_vec) -> None:
        self.id_string = id_string
        self.idx = None  # set when the Node is added to graph
        self.id_vec = id_vec
        self.row_vec = np.copy(self.id_vec)

    def __hash__(self) -> int:
        if self.idx is not None:
            return self.idx
        else:
            return str(self).__hash__()

    def __repr__(self):
        return self.id_string

    def __str__(self):
        return self.id_string


class HoloGraph(object):
    """A graph represented with high dimensional sparse vectors."""
    def __init__(self, edges, params) -> None:
        # read parameters from file, overwriting with keyword arguments
        self.params = params
        self.vector_model = vectors.VectorModel(self.params['DIM'],
                                                self.params['PERCENT_NON_ZERO'],
                                                self.params['BIND_OPERATION'])
        
        # Each token gets an int ID which specifies its index
        # in self.nodes and self.activations.
        self.string_to_index = OrderedDict()  # type: Dict[str, int]
        self.nodes = []
        self.edge_permutations = {edge: self.vector_model.permutation()
                                  for edge in edges}

    def create_node(self, id_string, id_vec=None) -> Node:
        """Returns a node."""
        if id_vec is None:
            id_vec = self.vector_model.sparse()
        return Node(self, id_string, id_vec)

    def add_node(self, node) -> None:
        """Adds a node to the graph."""
        idx = len(self.nodes)
        node.idx = idx
        self.string_to_index[str(node)] = idx
        self.nodes.append(node)

    def bump_edge(self, edge, node1, node2, factor) -> None:
        """Increases the weight of an edge from node1 to node2."""
        edge_vec = node2.id_vec[self.edge_permutations[edge]]
        node1.row_vec += factor * edge_vec

    def edge_weight(self, edge, node1, node2, generalize=0.0) -> float:
        """Returns the weight of an edge from node1 to node2"""
        row_vec = node1.row_vec
        if generalize:
            similarities = [vectors.cosine(node1.row_vec, node.row_vec)
                            for node in self.nodes]
            
            for n, s in zip(self.nodes, similarities):
                if s == np.nan:
                    print(n)

            #print('sim:', sum(similarities))
            gen_vec = sum(node.row_vec * similarity
                          for node, similarity in zip(self.nodes, similarities)
                          if similarity > .05)  # don't waste time on low effect computation
            row_vec += generalize * vectors.normalize(gen_vec)

        edge_vec = node2.id_vec[self.edge_permutations[edge]]
        return vectors.cosine(row_vec, edge_vec)

    def decay(self) -> None:
        """Decays all learned connections between nodes.

        This is done by adding a small factor of each nodes id_vec to
        its row vector, effectively making each node more similar
        to its initial state"""
        for node in self.nodes:
            node.row_vec += node.id_vec * self.params['DECAY_RATE']

    def distribution(self, node, edge, exp=1):
        """A statistical distribution defined by this nodes edges.

        This is used for introspection and `speak_markov` thus it
        is not part of the core of the model"""
        edges = [self.graph.edge_weight(edge, node, node2)
                 for node2 in self.nodes]

        distribution = (np.array(edges) + 1.0) / 2.0  # probabilites must be non-negative
        distribution **= exp  # accentuate differences
        return distribution / np.sum(distribution)

    def __getitem__(self, node_string) -> Node:
        try:
            idx = self.string_to_index[node_string]
            return self.nodes[idx]
        except KeyError:
            raise KeyError('{node_string} is not in the graph.'.format_map(locals()))

    def get(self, node_string, default=None):
        try:
            return self[node_string]
        except:
            return default

    def __contains__(self, node_string) -> bool:
        assert isinstance(node_string, str)
        return node_string in self.string_to_index
