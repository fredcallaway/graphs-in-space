from collections import Counter, defaultdict, OrderedDict
import utils


LOG = utils.get_logger(__name__, stream='INFO', file='WARNING')


class Node(object):
    """A node in a HoloGraph.

    Note that all HoloNodes must have a parent HoloGraph. However, they
    do not necessarily need to be in the graph as such.


    Attributes:
        string: e.g. [the [big dog]]
        idx: an int identifier
        id_vec: a random sparse vector that never changes
    """
    def __init__(self, graph, id_string) -> None:
        self.graph = graph
        self.id_string = id_string
        self.idx = None  # set when the node is added to graph

    def bump_edge(self, edge, node, factor) -> None:
        """Increases the weight of an edge to another node."""
        self.graph.edge_counts[edge][self.id_string][node.id_string] += factor

    def edge_weight(self, edge, node) -> float:
        """Returns the weight of an edge to another node.

        Between 0 and 1 inclusive.
        """
        edge_count = self.graph.edge_counts[edge][self.id_string][node.id_string]
        self_count = sum(self.graph.edge_counts[edge][self.id_string].values())
        if self_count == 0:
            return 0
        else:
            return edge_count / self_count

    def __hash__(self) -> int:
        if self.idx is not None:
            return self.idx
        else:
            return str(self).__hash__()

    def __repr__(self):
        return self.id_string

    def __str__(self):
        return self.id_string


class Graph(object):
    """A graph represented with high dimensional sparse vectors."""
    def __init__(self, edges, params) -> None:
        # read parameters from file, overwriting with keyword arguments
        self.params = params
        
        # Each token gets an int ID which specifies its index
        # in self.nodes and self.activations.
        self.string_to_index = OrderedDict()  # type: Dict[str, int]
        self.nodes = []

        self.edge_counts = {edge: defaultdict(Counter)
                            for edge in edges}

    def add_node(self, node) -> None:
        """Adds a node to the graph."""
        idx = len(self.nodes)
        node.idx = idx
        self.string_to_index[str(node)] = idx
        self.nodes.append(node)

    def decay(self) -> None:
        """Decays all learned connections between nodes.

        This is done by subtracting a small constant from each edge weight.
        Non-positive weighted edges are deleted."""
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

    def __getitem__(self, node_string) -> Node:
        try:
            idx = self.string_to_index[node_string]
            return self.nodes[idx]
        except KeyError:
            raise KeyError('{node_string} is not in the graph.'.format_map(locals()))
 
    def __contains__(self, node_string) -> bool:
        assert isinstance(node_string, str)
        return node_string in self.string_to_index
