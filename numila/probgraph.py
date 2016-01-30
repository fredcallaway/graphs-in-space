from collections import Counter, defaultdict, OrderedDict
import utils


from abstract_graph import MultiGraph


class ProbNode(object):
    """A node in a ProbGraph.

    Attributes:
        string: e.g. [the [big dog]]
    """
    def __init__(self, id_string) -> None:
        self.id_string = id_string

    def __hash__(self) -> int:
        return str(self).__hash__()

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
        self.params = params
        self.nodes = {}

        # Each edge is represented with a sparse matrix.
        self.edge_counts = {edge: defaultdict(Counter)
                            for edge in edges}

    def create_node(self, id_string):
        return ProbNode(id_string)

    def bind(self, node1, node2):
        id_string = '[{node1.id_string} {node2.id_string}]'.format_map(locals())
        return ProbNode(id_string)

    def bump_edge(self, edge, node1, node2, factor) -> None:
        self.edge_counts[edge][node1.id_string][node2.id_string] += factor

    def edge_weight(self, edge, node1, node2, verbose=False) -> float:
        edge_count = self.edge_counts[edge][node1.id_string][node2.id_string]
        self_count = sum(self.edge_counts[edge][node1.id_string].values())
        self_count += 10   # TODO unmagic
        if self_count == 0:
            return 0
        else:
            if verbose:
                print(edge_count, self_count)
            return edge_count / self_count

    def add_node(self, node) -> None:
        """Adds a node to the graph."""
        self.nodes[node.id_string] = node

    def decay(self) -> None:
        """Decays all learned connections between nodes."""
        for edge_type in self.edge_counts:
            for node1, edges in self.edge_counts[edge_type].items():
                for node2 in edges:
                    edges[node2] -= self.params['DECAY_RATE']
                
                non_pos_edges = [node2
                                 for node2, weight in edges.items()
                                 if weight <= 0]
                for node2 in non_pos_edges:
                    del edges[node2]

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
