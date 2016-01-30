from abc import ABCMeta, abstractmethod

class MultiGraph(metaclass=ABCMeta):
    """A graph with multiple types of edges and 0..1 bounded edge weights."""
    
    @abstractmethod
    def create_node(self, id_string):
        """Creates a node without adding it to the graph.

        The returned object can be treated just as any other node."""
        pass
    
    @abstractmethod
    def bind(self, node1, node2):
        """Returns a node representing the combination of two nodes."""
        pass
    
    @abstractmethod
    def add_node(self, node):
        """Adds the node to the graph, storing it for future use."""
        pass
    
    @abstractmethod
    def bump_edge(self, edge, node1, node2, factor) -> None:
        """Increases the weight of the given edge type between two nodes."""
        pass
    
    @abstractmethod
    def edge_weight(self, edge, node1, node2) -> float:
        """Returns the weight of the given edge type between two nodes.

        Between 0 and 1 inclusive.
        """
        pass

    @abstractmethod
    def decay(self):
        """Decays all learned connections between nodes."""
        pass

    def get(self, node_string, default=None):
        """Returns the node if it's in the graph, else `default`."""
        try:
            return self[node_string]
        except KeyError:
            return default
    
    @abstractmethod
    def __getitem__(self, node_string):
        pass
    
    @abstractmethod 
    def __contains__(self, node_string):
        pass
