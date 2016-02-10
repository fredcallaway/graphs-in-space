from collections import OrderedDict
from functools import lru_cache
from typing import Dict, List
import numpy as np

import utils
import vectors
from abstract_graph import MultiGraph

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
    def __init__(self, graph, id_string, id_vec=None):
        self.id_string = id_string
        if id_vec is not None:
            self.id_vec = id_vec
        else:
            self.id_vec = graph.vector_model.sparse()
        self.row_vec = graph.vector_model.sparse()
        self._original_row = np.copy(self.row_vec)

    #def __hash__(self):
    #    return hash(self.id_string)

    def __repr__(self):
        return self.id_string

    def __str__(self):
        return self.id_string


class HoloGraph(MultiGraph):
    """A graph represented with high dimensional sparse vectors."""
    def __init__(self, edges, params):
        # read parameters from file, overwriting with keyword arguments
        self.params = params
        self.vector_model = vectors.VectorModel(self.params['DIM'],
                                                self.params['PERCENT_NON_ZERO'],
                                                self.params['BIND_OPERATION'])
        
        self.nodes = {}
        self.edge_permutations = {edge: self.vector_model.permutation()
                                  for edge in edges}

    def create_node(self, id_string):
        return HoloNode(self, id_string)

    def bind(self, node1, node2, edges=None) -> HoloNode:
        id_string = '[{node1.id_string} {node2.id_string}]'.format_map(locals())
        id_vec = self.vector_model.bind(node1.id_vec, node2.id_vec)
        return HoloNode(self, id_string, id_vec=id_vec)

    def add_node(self, node):
        """Adds a node to the graph."""
        assert isinstance(node, HoloNode)
        self.nodes[node.id_string] = node

    def bump_edge(self, edge, node1, node2, factor) -> None:
        """Increases the weight of an edge to another node."""
        edge_vec = node2.id_vec[self.edge_permutations[edge]]
        node1.row_vec += factor * edge_vec
        
        #self.graph._edge_counts[edge][self.id_string][node.id_string] += 1
        self.edge_weight.cache_clear()

    @lru_cache(maxsize=None)
    def edge_weight(self, edge, node1, node2) -> float:
        """Returns the weight of an edge to another node.

        Between 0 and 1 inclusive.
        """
        edge_vec = node2.id_vec[self.edge_permutations[edge]]
        weight = vectors.cosine(node1.row_vec, edge_vec)
        if weight < 0:
            return 0.0
        else:
            return weight

    def decay(self):
        """Decays all learned connections between nodes.

        This is done by adding a small factor of each nodes id_vec to
        its row vector, effectively making each node more similar
        to its initial state"""
        decay = self.params['DECAY']
        if not decay:
            return
        for node in self.nodes.values():
            node.row_vec += node._original_row * decay

    def get(self, node_string, default=None):
        """Returns the node if it's in the graph, else `default`."""
        try:
            return self[node_string]
        except KeyError:
            return default

    def __getitem__(self, node_string):
        try:
            return self.nodes[node_string]
        except KeyError:
            raise KeyError('{node_string} is not in the graph.'.format_map(locals()))
 
    def __contains__(self, node_string):
        assert isinstance(node_string, str)
        return node_string in self.nodes