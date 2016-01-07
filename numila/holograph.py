from collections import Counter, defaultdict, OrderedDict
from functools import lru_cache
import itertools
from typing import Dict, List
import numpy as np
import utils
import vectors


LOG = utils.get_logger(__name__, stream='INFO', file='WARNING')

COUNT = 0

class HoloNode(object):
    """A node in a HoloGraph.

    Note that all HoloNodes must have a parent HoloGraph. However, they
    do not necessarily need to be in the graph as such.

    Attributes:
        string: e.g. [the [big dog]]
        idx: an int identifier
        id_vec: a random sparse vector that never changes
    """
    def __init__(self, graph, id_string, id_vec=None) -> None:
        self.graph = graph
        self.id_string = id_string
        self.idx = None  # set when the node is added to graph
        if id_vec is not None:
            self.id_vec = id_vec
        else:
            self.id_vec = self.graph.vector_model.sparse()
        self.row_vec = np.copy(self.id_vec)

    def bump_edge(self, edge, node, factor) -> None:
        """Increases the weight of an edge to another node."""
        edge_vec = node.id_vec[self.graph.edge_permutations[edge]]
        self.row_vec += factor * edge_vec
        
        self.graph._edge_counts[edge][self.id_string][node.id_string] += 1
        self.edge_weight.cache_clear()

    @lru_cache(maxsize=None)
    def edge_weight(self, edge, node) -> float:
        """Returns the weight of an edge to another node.

        Between 0 and 1 inclusive.
        """
        edge_vec = node.id_vec[self.graph.edge_permutations[edge]]
        weight = vectors.cosine(self.row_vec, edge_vec)
        if weight < 0:
            return 0.0
        else:
            return weight

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
    Node = HoloNode

    def __init__(self, edges, params):
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

        self._edge_counts = {edge: defaultdict(Counter)
                             for edge in edges}

    def add_node(self, node):
        """Adds a node to the graph."""
        assert isinstance(node, self.Node)
        idx = len(self.nodes)
        node.idx = idx
        self.string_to_index[str(node)] = idx
        self.nodes.append(node)

    def decay(self):
        """Decays all learned connections between nodes.

        This is done by adding a small factor of each nodes id_vec to
        its row vector, effectively making each node more similar
        to its initial state"""
        for node in self.nodes:
            node.row_vec += node.id_vec * self.params['DECAY_RATE']

    def get(self, node_string, default=None):
        """Returns the node if it's in the graph, else `default`."""
        try:
            return self[node_string]
        except KeyError:
            return default

    def __getitem__(self, node_string):
        try:
            idx = self.string_to_index[node_string]
            return self.nodes[idx]
        except KeyError:
            raise KeyError('{node_string} is not in the graph.'.format_map(locals()))
 
    def __contains__(self, node_string):
        assert isinstance(node_string, str)
        return node_string in self.string_to_index
