from collections import OrderedDict
from functools import lru_cache
from typing import Dict, List
import numpy as np
from scipy import stats

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
    def __init__(self, graph, id_string, children=(), row_vec=None):
        super().__init__(graph, id_string, children)
        params = self.graph.params
        self.id_vec = graph.vector_model.sparse()

        if params['DYNAMIC']:
            self.dynamic_vec = self.graph.vector_model.sparse()
            #self.dynamic_vec = np.ones(params['DIM'])

        self.row_vec = row_vec if row_vec is not None else graph.vector_model.sparse()
        assert not np.isnan(np.sum(self.row_vec))
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

    def bind(self, *nodes, edges=None, composition=False):
        id_string = self._id_string(nodes)
        
        if self.params['COMPOSITION']:
            # gen_vec is the weighted average of all other same-length blobs
            gen_vec = self.vector_model.zeros()
            comparable = (n for n in self.nodes if len(n.children) == len(nodes))
            for blob in comparable:
                similarity = stats.gmean([vectors.cosine(n.row_vec, c.row_vec)
                                         for n, c in zip(nodes, blob.children)])
                gen_vec += similarity * vectors.normalize(blob.row_vec)

            gen_vec = self.params['COMPOSITION'] * vectors.normalize(gen_vec)
            row_vec = self.vector_model.sparse() + gen_vec
            if np.isnan(np.sum(row_vec)):
                import IPython; IPython.embed()

        else:
            row_vec = None

        return HoloNode(self, id_string, children=nodes, row_vec=row_vec)

    def sum(self, nodes, weights=None):
        weights = list(weights)
        if weights:
            ids = [n.id_vec * w for n, w in zip(nodes, weights)]
            rows = [n.row_vec * w for n, w in zip(nodes, weights)]
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
