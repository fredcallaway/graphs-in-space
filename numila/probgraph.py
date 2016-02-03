from collections import Counter, defaultdict
import utils

from abstract_graph import MultiGraph


class ProbNode(object):
    """A node in a ProbGraph.

    Attributes:
        string: e.g. [the [big dog]]
    """
    def __init__(self, id_string, edges, edge_counts=None) -> None:
        self.id_string = id_string
        self.edge_counts = edge_counts or {edge: Counter() for edge in edges}

    def bump_edge(self, edge, node, factor) -> None:
        self.edge_counts[edge][node.id_string] +=  factor

    def edge_weight(self, edge, node) -> float:
        edge_count = self.edge_counts[edge][node.id_string]
        self_count = sum(self.edge_counts[edge].values())
        if self_count == 0:
            return 0
        else:
            return edge_count / self_count

    def __hash__(self) -> int:
        return hash(self.id_string)

    def __repr__(self):
        return self.id_string

    def __str__(self):
        return self.id_string


class ProbGraph(MultiGraph):
    """A graph where edges represent conditional probabilities.

    Nodes represent entities and edge types represent relations. Weights
    on edges represent the probability of the target node being in the
    given relationship with the source node.

    A.edge_weight(R, B) is the probability that B is in relation R
    with B given that A has occurred. For example, if R represents
    temporal precedence, then A.edge_weight(R, B) would be the probability
    that B has previously occurred given that A just occurred.
    """
    def __init__(self, edges, params) -> None:
        self.edges = edges
        self.params = params
        self.nodes = {}

    def create_node(self, id_string) -> ProbNode:
        return ProbNode(id_string, self.edges)

    def bind(self, node1, node2, edges={}) -> ProbNode:
        id_string = '[{node1.id_string} {node2.id_string}]'.format_map(locals())
        edge_counts = {edge: node.edge_counts[edge] for edge, node in edges.items()}
        return ProbNode(id_string, self.edges, edge_counts)

    def bump_edge(self, edge, node1, node2, factor) -> None:
        node1.bump_edge(edge, node2, factor)

    def edge_weight(self, edge, node1, node2) -> float:
        return node1.edge_weight(edge, node2)

    def add_node(self, node) -> None:
        """Adds a node to the graph."""
        self.nodes[node.id_string] = node

    def decay(self) -> None:
        """Decays all learned connections between nodes."""
        for node1 in self.nodes.values():
            for edge_type, counter in node1.edge_counts.items():
                # Decay every edge of this type out of node1.
                for node2 in counter:
                    counter[node2] -= self.params['DECAY_RATE']
                # Delete non-positive edges.
                non_pos = [node for node, weight in counter.items()
                           if weight <= 0]
                for node in non_pos:
                    del counter[node]

        #for edge_type in self.edge_counts:
        #    for node1, edges in self.edge_counts[edge_type].items():
        #        for node2 in edges:
        #            edges[node2] -= self.params['DECAY_RATE']
                
        #        non_pos_edges = [node2
        #                         for node2, weight in edges.items()
        #                         if weight <= 0]
        #        for node2 in non_pos_edges:
        #            del edges[node2]

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
