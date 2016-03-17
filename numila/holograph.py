from collections import OrderedDict
from functools import lru_cache
from typing import Dict, List
import numpy as np

import utils
import vectors
from abstract_graph import HiGraph, HiNode

LOG = utils.get_logger(__name__, stream='INFO', file='WARNING')

COUNT = 0


class HoloNode(HiNode):
    """A node in a HoloGraph.

    Note that all HoloNodes must have a parent HoloGraph. However, they
    do not necessarily need to be in the graph as such.

    Attributes:
        string: e.g. [the [big dog]]
        idx: an int identifier
        id_vec: a random sparse vector that never changes
    """
    def __init__(self, graph, id_string, children=(), id_vec=None, row_vec=None):
        super().__init__(graph, id_string, children)
        params = self.graph.params
        
        if False and params['OLD_DYNAMIC']:  # TODO
            # Use a dynamic id_vec for generalization.
            id_vec = np.zeros(params['DIM'])
            static_len = (1 - params['DYNAMIC']) * params['DIM']
            static_vec = graph.vector_model.sparse(static_len)
            id_vec[:static_len] = static_vec

            # These vectors are all pointers to the same array.
            self.id_vec = id_vec
            self.static_vec = id_vec[:static_len]
            self.dynamic_vec = id_vec[static_len:]
        else:
            self.id_vec = id_vec if id_vec is not None else graph.vector_model.sparse()

        if params['DYNAMIC']:
            self.dynamic_vec = self.graph.vector_model.sparse()
            #self.dynamic_vec = np.ones(params['DIM'])

        self.row_vec = row_vec if row_vec is not None else graph.vector_model.sparse()
        self._original_row = np.copy(self.row_vec)

    def bump_edge(self, node, edge, factor=1):
        """Increases the weight of an edge to another node."""
        # Add other node's id_vec to this node's row_vec
        edge_vec = node.id_vec[self.graph.edge_permutations[edge]]
        self.row_vec += factor * edge_vec
        self.edge_weight.cache_clear()
        
        if False and self.graph.params['OLD_DYNAMIC']:  # TODO
            # Add this node's row_vec to other node's id_vec
            compressed_row_vec = vectors.compress(self.row_vec, len(node.dynamic_vec))
            node.dynamic_vec += compressed_row_vec

        if self.graph.params['DYNAMIC']:
            node.dynamic_vec += self.row_vec * factor
            self.row_vec += (vectors.normalize(node.dynamic_vec) 
                             * factor * self.graph.params['DYNAMIC'])



    @lru_cache(maxsize=None)
    def edge_weight(self, node, edge):
        """Returns the weight of an edge to another node.

        Between 0 and 1 inclusive.
        """
        edge_vec = node.id_vec[self.graph.edge_permutations[edge]]
        weight = vectors.cosine(self.row_vec, edge_vec)
        if weight < 0:
            return 0.0
        else:
            return weight

    def similarity(self, node):
        return vectors.cosine(self.row_vec, node.row_vec)


class HoloGraph(HiGraph):
    """A graph represented with high dimensional sparse vectors."""
    def __init__(self, edges, params):
        # read parameters from file, overwriting with keyword arguments
        self.params = params
        self.vector_model = vectors.VectorModel(self.params['DIM'],
                                                self.params['PERCENT_NON_ZERO'],
                                                self.params['BIND_OPERATION'])
        
        self.edge_permutations = {edge: self.vector_model.permutation()
                                  for edge in edges}
        self._nodes = {}

    def create_node(self, id_string):
        return HoloNode(self, id_string)

    def bind(self, node1, node2, edges=None, composition=False):
        id_string = '[{node1.id_string} {node2.id_string}]'.format_map(locals())

        if self.params['COMPOSITION']:
            # Create id_vec by binding the children id vectors.
            id_vec = self.vector_model.bind(node1.id_vec, node2.id_vec)
            comp_vec = self.vector_model.bind(node1.row_vec, node2.row_vec)
            
            # Create a row vector based on other chunks.
            row_vec = self.vector_model.sparse()
            for blob in (node for node in self.nodes if node.children):
                c1, c2 = blob.children
                blob_comp_vec = self.vector_model.bind(c1.row_vec, c2.row_vec)
                similarity = vectors.cosine(comp_vec, blob_comp_vec)
                row_vec += (similarity * blob.row_vec) * self.params['COMPOSITION']
        
        elif self.params['COMP2']:
            row_vec = self.vector_model.sparse()
            for blob in (node for node in self.nodes if node.children):
                c1, c2 = blob.children
                sim1 = vectors.cosine(node1, c1)
                sim2 = vectors.cosine(node2, c2)
                full_sim = sim1 * sim2
                row_vec += (full_sim * self.params['COMPOSISION'] * vectors.normalize(blob.row_vec))
            id_vec = None

        else:
            row_vec = None
            id_vec = None

        return HoloNode(self, id_string, children=(node1, node2),
                        id_vec=id_vec, row_vec=row_vec)

    def sum(self, nodes, weights=None):
        weights = list(weights)
        if weights:
            ids = [n.id_vec * w for n, w in zip(self.nodes, weights)]
            rows = [n.row_vec * w for n, w in zip(self.nodes, weights)]
        else:
            ids = [n.id_vec for n in nodes]
            rows = [n.row_vec for n in nodes]
        
        id_vec = np.sum(ids, axis=0)
        row_vec = np.sum(rows, axis=0)
        node =  HoloNode(self, '__SUM__', id_vec, row_vec)
        return node

    def decay(self):
        """Decays all learned connections between nodes.

        This is done by adding a small factor of each nodes id_vec to
        its row vector, effectively making each node more similar
        to its initial state"""
        decay = self.params['DECAY']
        if not decay:
            return
        for node in self.nodes:
            node.row_vec += node._original_row * decay

if __name__ == '__main__':
    import pytest
    pytest.main(['test_graph.py'])
