from collections import Counter, defaultdict
import utils
import itertools

from abstract_graph import HiGraph, HiNode


class ProbNode(HiNode):
    """A node in a ProbGraph.

    Attributes:
        string: e.g. [the [big dog]]
    """
    def __init__(self, graph, id_string, edges, children=()) -> None:
        super().__init__(graph, id_string, children)
        if isinstance(edges, dict):
            self.edge_counts = edges
        else:
            self.edge_counts = {edge: Counter() for edge in edges}

    def bump_edge(self, node, edge, factor=1) -> None:
        self.edge_counts[edge][node.id_string] +=  factor

    #@utils.contract(lambda x: 0 <= x <= 1)
    def edge_weight(self, node, edge) -> float:
        edge_count = self.edge_counts[edge][node.id_string]
        self_count = sum(self.edge_counts[edge].values())
        if edge_count == 0:
            return 0.0
        else:
            return edge_count / self_count

    def similarity(self, node):
        return 0.0


class ProbGraph(HiGraph):
    """A graph where edges represent conditional probabilities.

    Nodes represent entities and edge types represent relations. Weights
    on edges represent the probability of the target node being in the
    given relationship with the source node.

    A.edge_weight(R, B) is the probability that B is in relation R
    with B given that A has occurred. For example, if R represents
    temporal precedence, then A.edge_weight(R, B) would be the probability
    that B has previously occurred given that A just occurred.
    """
    def __init__(self, edges, DECAY=False, HIERARCHICAL=True, **kwargs) -> None:
        # TODO: kwargs is just so that we can pass more parameters than are
        # actually used.
        super().__init__()
        self.edges = edges
        self.DECAY = DECAY
        self.HIERARCHICAL = HIERARCHICAL

    def create_node(self, id_string) -> ProbNode:
        return ProbNode(self, id_string, self.edges)

    def bind(self, *nodes, edges=None) -> ProbNode:
        if self.HIERARCHICAL:
            children = nodes
        else:
            children = self._concatenate_children(nodes)

        id_string = self._id_string(children)
        return ProbNode(self, id_string, self.edges, children=children)


    def decay(self) -> None:
        """Decays all learned connections between nodes."""
        decay = self.DECAY
        if not decay:
            return
        for node1 in self.nodes:
            for edge_type, counter in node1.edge_counts.items():
                # Decay every edge of this type out of node1.rs
                for node2 in counter:
                    counter[node2] -= decay
                # Delete non-positive edges.
                non_pos = [node for node, weight in counter.items()
                           if weight <= 0]
                for node in non_pos:
                    del counter[node]
