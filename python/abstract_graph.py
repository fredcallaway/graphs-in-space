from abc import ABCMeta, abstractmethod

class HiBlob(object):
    pass
    # should we just have default None for child1 and child2 ?

class HiNode(metaclass=ABCMeta):
    """A node in a HiGraph.

    All HiNodes must have a parent HiGraph. However, they
    do not necessarily need to be in the graph as such.
    """
    def __init__(self, graph, id_string, children=()):
        self.graph = graph
        self.id_string = id_string
        self.children = tuple(children)

    @property
    def child1(self):
        return self.children[0] if self.children else None

    @property
    def child2(self):
        return self.children[1] if self.children else None

    @abstractmethod
    def bump_edge(self, node, edge, factor=1):
        """Increases the weight of the given edge type to another node."""
        pass

    @abstractmethod
    def edge_weight(self, node, edge):
        """Returns the weight of the given edge type to another node.

        Between 0 and 1 inclusive.
        """
        pass

    @abstractmethod
    def similarity(self, node):
        """Returns similarity to another node.

        Between 0 and 1 inclusive."""
        pass

    def __repr__(self):
        return self.id_string

    def __str__(self):
        return self.id_string

    def __len__(self):
        if self.children:
            return len(self.children)
        else:
            return 1

class HiGraph(metaclass=ABCMeta):
    """A graph with multiple types of edges and 0..1 bounded edge weights."""
    def __init__(self):
        self._nodes = {}

        self.MIN_COUNT = 2 # TODO

    @property
    def nodes(self):
        return self._nodes.values()

    def add(self, node):
        """Adds a node to the graph."""
        if isinstance(node, str):
            id_string = node
            node = self.create_node(id_string)
        
        elif node.children:
            for child in node.children:
                assert child in self
                if hasattr(child, 'count') and child.count < self.MIN_COUNT:
                    return  # TODO

        self._nodes[node.id_string] = node
    
    @abstractmethod
    def create_node(self, id_string):
        """Creates a node without adding it to the graph.

        The returned object can be treated just as any other node."""
        pass
    
    @abstractmethod
    def bind(self, *nodes):
        """Returns a blob, a new node representing a list of nodes."""
        pass

    @abstractmethod
    def decay(self):
        """Decays all learned connections between nodes."""
        pass

    def get(self, id_string, default=None, add=False):
        """Returns the node if it's in the graph, else `default`."""
        try:
            return self._nodes[id_string]
        except KeyError:
            if add:
                new_node = self.create_node(id_string)
                self.add(new_node)
                return new_node
            else:
                return default

    def get_chunk(self, *nodes):
        id_string = self._id_string(nodes)
        return self.get(id_string)

    @staticmethod
    def _id_string(nodes):
        # e.g. [A B C]
        return '[' + ' '.join(node.id_string for node in nodes) + ']'

    @staticmethod
    def _concatenate_children(nodes):
        children = []
        for node in nodes:
            if node.children:
                children.extend(node.children)
            else:
                children.append(node)
        return children

    def __len__(self):
        return len(self._nodes)

    def __getitem__(self, node_string):
        try:
            return self._nodes[node_string]
        except KeyError:
            raise KeyError('{node_string} is not in the graph.'.format_map(locals()))
    
    def __contains__(self, node):
        if isinstance(node, str):
            return node in self._nodes
        elif isinstance(node, HiNode):
            return self._nodes.get(node.id_string) is node
        else:
            raise ValueError('Can only check if strings and HiNodes are in the graph.')

