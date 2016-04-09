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
    def __init__(self, graph, id_string, children=(), row_vecs={}):
        super().__init__(graph, id_string, children)
        self.id_vec = graph.vector_model.sparse()

        #self.row_vec = row_vec if row_vec is not None else graph.vector_model.sparse()
        self.row_vecs = {edge: graph.vector_model.sparse()
                         for edge in graph.edges}
        self.row_vecs.update(row_vecs)

        #self._original_rows = np.copy(self.row_vec)

        if self.graph.DYNAMIC:
            self.dynamic_vecs = {edge: graph.vector_model.sparse()
                                 for edge in graph.edges}
            self.gen_vecs = {edge: np.copy(vec)
                             for edge, vec in self.row_vecs.items()}


    def bump_edge(self, node, edge, factor=1):
        """Increases the weight of an edge to another node."""
        
        # Add other node's id_vec to this node's row_vec
        #edge_vec = node.id_vec[self.graph.edge_permutations[edge]]
        #self.row_vec += factor * edge_vec
        
        self.row_vecs[edge] += node.id_vec * factor        

        if self.graph.DYNAMIC:
            # The target node learns that this node points to it.
            node.dynamic_vecs[edge] += self.row_vecs[edge] * factor
            # This node's generalized vectors point to nodes that 
            # other nodes that point to target node point to.
            self.gen_vecs[edge] += (vectors.normalize(node.dynamic_vecs[edge]) 
                                    * (self.graph.vector_model.magnitude
                                       * factor * self.graph.DYNAMIC))
        self.edge_weight.cache_clear()


    @lru_cache(maxsize=None)
    @utils.contract(lambda x: 0 <= x <= 1)
    def edge_weight(self, node, edge, generalize=False):
        """Returns the weight of an edge to another node.

        Between 0 and 1 inclusive.
        """
        #edge_vec = node.id_vec[self.graph.edge_permutations[edge]]
        #weight = vectors.cosine(self.row_vec, edge_vec)
        self_vec = (self.gen_vecs if generalize else self.row_vecs)[edge]
        cos = vectors.cosine(self_vec, node.id_vec)
        return max(cos, 0.0)

    def similarity(self, node, weights=None):
        """Weighted geometric mean of cosine similarities for each edge."""
        weights = weights or np.ones(len(self.graph.edges))
        assert len(weights) == len(self.graph.edges)

        edge_sims = [vectors.cosine(self.row_vecs[edge], node.row_vecs[edge])
                     ** weight
                     for edge, weight in zip(self.graph.edges, weights)]
        return stats.gmean(edge_sims)


class HoloGraph(HiGraph):
    """A graph represented with high dimensional sparse vectors."""
    def __init__(self, edges, DIM=10000, PERCENT_NON_ZERO=0.005, 
                 BIND_OPERATION='addition', HIERARCHICAL=True, 
                 COMPOSITION=False, DECAY=0, DYNAMIC=0, **kwargs):
        # TODO: kwargs is just so that we can pass more parameters than are
        # actually used.
        super().__init__()
        self.edges = edges
        self.DYNAMIC = DYNAMIC
        self.DIM = DIM
        self.PERCENT_NON_ZERO = PERCENT_NON_ZERO
        self.BIND_OPERATION = BIND_OPERATION
        self.HIERARCHICAL = HIERARCHICAL
        self.COMPOSITION = COMPOSITION
        self.DECAY = DECAY
        self.vector_model = vectors.VectorModel(self.DIM,
                                                self.PERCENT_NON_ZERO,
                                                self.BIND_OPERATION)
        
        #self.edge_permutations = {edge: self.vector_model.permutation()
        #                          for edge in edges}

    def create_node(self, id_string):
        return HoloNode(self, id_string)

    def bind(self, *nodes, composition=False):
        if self.HIERARCHICAL:
            children = nodes
        else:
            children = self._concatenate_children(nodes)

        if self.COMPOSITION:
            # gen_vec is the weighted average of all other same-length blobs
            gen_vec = self.vector_model.zeros()
            comparable = (n for n in self.nodes if len(n.children) == len(children))
            for blob in comparable:
                similarity = stats.gmean([vectors.cosine(n.row_vec, c.row_vec)
                                         for n, c in zip(children, blob.children)])
                gen_vec += similarity * vectors.normalize(blob.row_vec)

            gen_vec = self.COMPOSITION * vectors.normalize(gen_vec)
            row_vec = self.vector_model.sparse() + gen_vec
            if np.isnan(np.sum(row_vec)):
                import IPython; IPython.embed()

        else:
            row_vec = None

        id_string = self._id_string(children)
        # TODO should be children=children ?
        return HoloNode(self, id_string, children=nodes, row_vec=row_vec)

    def sum(self, nodes, weights=None):
        weights = weights and list(weights)
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
        """Decays all learned connections between nodes."""
        assert False, 'unimplimented'
        decay = self.DECAY
        if not decay:
            return
        for node in self.nodes:
            node.row_vec += node._original_row * decay


if __name__ == '__main__':
    import pytest
    pytest.main(['test_graph.py'])
