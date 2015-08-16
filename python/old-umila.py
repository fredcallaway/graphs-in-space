from __future__ import division
import numpy as np
from pprint import pprint
from collections import OrderedDict

class Model(object):
    """A model that learns hierarchical chunk-based representations of sequences

    Parameters:
      max_chunk_size (int): the maximum number of words that can be in a chunk.
        This shouldn't exist, but is necessary for my node finding function.
    """
    def __init__(self, max_chunk_size, activation_threshold):
        super(Model, self).__init__()
        self.max_chunk_size = max_chunk_size 
        self.activation_threshold = activation_threshold

        self.graph = Higraph(self)
        self.memory = Memory(self)

    def read(self, token):
        """Reads one token and performs all associated computation"""
        self.memory.add(token)
        self.graph.decay()

    def read_file(self, file_path):
        with open(file_path) as f:
            for line in f:
                tokens = line.split(' ')
                del tokens[-1] # remove trailing \n
                for t in tokens:
                    self.read(t)

    def parse(self, utterance):
        """Returns all possible chunkings the utterance with probabilites.

        A "chunking" is a Chunk object that contains only Nodes, and has
        the utterance as a surface representation. Probability of a chunking
        is the probability of the first token given an utterance boundary
        times the product of the transitional proabability between each node.
        """
        # this is a work in progress
        utterance = utterance.split(' ')

        def recurse(previous_node, to_parse):
            if to_parse == '':
                return ([], [1])  # TODO: use probability given utterance boundary
            
            # this currently only allows one token Nodes, need to add a for loop
            first, rest = ' '.join(to_parse[:1]), ' '.join(to_parse[1:])
            nodes = self.graph[first]
            parses = []
            probabilities = []
            for node in nodes:  # all possible ways to pull of the next chunk
                if previous_node:
                    probability = self.graph.ftp(previous_node, node)
                else: # the beginning of an utterance
                    probability = node.weight

                # all possible ways to parse the remainder
                rest_parsed_options, rest_prob_options = recurse(node, rest)
                for parse, prob in zip(rest_parsed_options, rest_prob_options): 
                    # combine the pulled off chunk with rest
                    parses.append(Chunk(node, parse))
                    probabilities.append(probability * prob)

            return parses, probabilities
        
        return recurse(None, utterance)


class Memory(object):
    """The working memory of the Model.

    """
    def __init__(self, model):
        super(Memory, self).__init__()
        self.model = model
        self.activation_threshold = model.activation_threshold
        self.tokens = []
        self.activations = []
        self.match = []

    def add(self, token):
        self.tokens.append(token)
        self.activations.append(1.0)
        self.update_temporal_edges()
        self.decay()

    def decay(self):
        # TODO: look up exponential decay parameter
        self.activations = [a * 0.9 for a in self.activations]
        # remove tokens with activations below threshhold
        new_tokens = []
        new_activations = []
        for t, a in zip(self.tokens, self.activations):
            if a > self.activation_threshold:
                new_tokens.append(t)
                new_activations.append(a)
        self.tokens = new_tokens
        self.activations = new_activations

    def update_temporal_edges(self):
        """Updates temporal edges for nodes completed with most recent token."""
        for size in xrange(1, self.model.max_chunk_size + 1):
            terminating_nodes = self.get_terminating_nodes(-1, size)
            for node in terminating_nodes:
                node.bump_weight()
            previous_nodes = self.get_terminating_nodes(-1 - size)  # all sizes
            # bump egdes of all pre/post combinations
            for pre_node in previous_nodes:
                for post_node in terminating_nodes:
                    self.model.graph.bump_temporal_edge(pre_node, post_node)

    def get_terminating_nodes(self, position, size=None):
        """Returns set of all nodes terminating at `position` with length `size`."""
        if size:
            onset = (position + 1) - size
            if onset < - len(self.tokens):
                # can't have larger size than the number of tokens in memory!
                return {}
            offset = (position + 1)
            if offset == 0:
                # we want to go through the whole string
                offset = None
            surface_representation = ' '.join(self.tokens[onset: offset])
            terminating_nodes = self.model.graph[surface_representation]
            return terminating_nodes
        else:
            # return nodes of all possible sizes
            terminating_nodes = set()
            for size in xrange(1, self.model.max_chunk_size + 1):
                this_size_nodes = self.get_terminating_nodes(position, size=size)
                terminating_nodes = terminating_nodes.union(this_size_nodes)
            return terminating_nodes


class Higraph(OrderedDict):
    """Dict from surface representations to sets of Nodes"""
    def __init__(self, model):
        super(Higraph, self).__init__()
        self.model = model
        self.minimum_occurences = model.minimum_occurences
        self.node_prior = model.node_prior
        # self.collocation_matrix[0][1] is count of node 1 following node 0
        self.collocation_matrix = np.zeros((100000, 10000)) # TODO: make this dynamically expand
        self.weights = np.zeros(100000)
        self._next_id = 0
    
    def nodes(self):
        return self.values()

    def create_node(self, chunk):
        node = Node(chunk, self)
        surface_string = node.surface
        if surface_string in self:
            # there is an existing node with the same surface string
            # so we add the new node to the set
            self[surface_string].add(node)
        else:
            self[surface_string] = {node}

    def bump_temporal_edge(self, pre_node, post_node):
        pre_id, post_id = pre_node.id, post_node.id
        self.collocation_matrix[pre_id][post_id] += 1
        c = Chunk(pre_node, post_node)
        if (c not in self
            and min(pre_node.weight, post_node.weight) > self.minimum_occurences
            and self.barlows(pre_node, post_node)):
                self.create_node(Chunk(pre_node, post_node))

    def ftp(self, pre_node, post_node):
        """Probability of post_node following pre_node"""
        pre_id, post_id = pre_node.id, post_node.id
        this_node_collocations = self.collocation_matrix[pre_id, post_id]
        all_collocations = np.sum(self.collocation_matrix[pre_id])
        return this_node_collocations / all_collocations

    def btp(self, pre_node, post_node):
        """Probability of pre_node having preceded post_node"""
        pre_id, post_id = pre_node.id, post_node.id
        this_node_collocations = self.collocation_matrix[pre_id, post_id]
        all_collocations = np.sum(self.collocation_matrix[:, post_id])
        return this_node_collocations / all_collocations

    def barlows(self, pre_node, post_node):
        pre_id, post_id = pre_node.id, post_node.id
        pre_post = self.collocation_matrix[pre_id, post_id]  # pre => post
        pre_all = np.sum(self.collocation_matrix[pre_id, :])  # all forward collocations
        all_post = np.sum(self.collocation_matrix[:, post_id])  # all backward collocations

        barlow = pre_post / (pre_all * all_post)

        return True if barlow > self.node_prior else False

    def decay(self):
        """All weights decay"""
        pass

    @property
    def next_id(self):
        return self._next_id
        self._next_id += 1

    def __missing__(self, key):
        # add base token graph if not yet encountered
        if type(key) is int:
            # get node by id
            return self.nodes()[key]  # remember it's an OrderedDict
        elif isinstance(key, str):
            if ' ' not in key:
                # key is a base token, so we add it to the graph
                node = Node(Chunk(key), self)
                self[key] = {node}
                return {node}
            else:
                # there are no Nodes with that surface representation
                return set()

    def __contains__(self, arg):
        if isinstance(arg, str):
            return super(Higraph, self).__contains__(arg)
        elif isinstance(arg, Chunk):
            # TODO
            pass
        else:
            raise ValueError()


class Chunk(list):
    """The basic grammatical unit

    A list of Chunks or a singleton list of a base token
    """
    def __init__(self, *args):
        """
        Args:
          *args: a tuple of Chunks, or a single string representing a base token
        """
        super(Chunk, self).__init__()
        self.leaves = []
        for arg in args:
            self.append(arg)
            # update leaves
            if isinstance(arg, str):  # base token is a str
                assert len(args) is 1  # only singleton lists of base tokens allowed
                self.leaves.append(arg)
            elif isinstance(arg, Chunk):
                self.leaves.extend(arg.leaves)

    @property
    def surface(self):
        return ' '.join(self.leaves)

    def __repr__(self):
        if len(self) > 1:
            return '[' + ' '.join(map(str, self)) + ']'
        else:
            return str(self[0])

    # TODO: 
    #   slicing ?


class Node(Chunk):
    """A Chunk in a Higraph"""
    def __init__(self, chunk, graph):
        super(Node, self).__init__(*chunk)  # copies the structure of the chunk
        # self = chunk
        self.graph = graph
        self.id = graph.next_id
        self.leaves = chunk.leaves

        self.graph.weights[self.id] = 1
        self.similarity = {}

    @property
    def weight(self):
        return self.graph.weights[self.id]

    def bump_weight(self):
        self.graph.weights[self.id] += 1

    def __hash__(self):
        return self.id

    # def __repr__(self):
    #     return '(%s %s)' % (str(self.weight) , super(Node, self).__repr__())


class Slot(Node):
    """A slot that can be filled with a node

    Attributes:
      fillers (dict): a mapping of possible filler Nodes to their probabilities"""
    def __init__(self):
        super(Slot, self).__init__('___')
        self.fillers = {}


class Phonoloop(object):
    """The working memory of the Model.

    """
    def __init__(self, model):
        super(Memory, self).__init__()
        self.model = model
        self.activation_threshold = model.activation_threshold
        self.tokens = []
        self.activations = []
        self.match = []

    def add(self, token):
        self.tokens.append(token)
        self.activations.append(1.0)
        self.update_temporal_edges()
        self.decay()

    def decay(self):
        # TODO: look up exponential decay parameter
        self.activations = [a * 0.9 for a in self.activations]
        # remove tokens with activations below threshhold
        new_tokens = []
        new_activations = []
        for t, a in zip(self.tokens, self.activations):
            if a > self.activation_threshold:
                new_tokens.append(t)
                new_activations.append(a)
        self.tokens = new_tokens
        self.activations = new_activations

    def update_temporal_edges(self):
        """Updates temporal edges for nodes completed with most recent token."""
        for size in xrange(1, self.model.max_node_size + 1):
            terminating_nodes = self.get_terminating_nodes(-1, size)
            for node in terminating_nodes:
                node.bump_weight()
            previous_nodes = self.get_terminating_nodes(-1 - size)  # all sizes
            # bump egdes of all pre/post combinations
            for pre_node in previous_nodes:
                for post_node in terminating_nodes:
                    self.model.graph.bump_temporal_edge(pre_node, post_node)

    def get_terminating_nodes(self, position, size=None):
        """Returns set of all nodes terminating at `position` with length `size`."""
        if size:
            onset = (position + 1) - size
            if onset < - len(self.tokens):
                # can't have larger size than the number of tokens in memory!
                return {}
            offset = (position + 1)
            if offset == 0:
                # we want to go through the whole string
                offset = None
            surface_representation = ' '.join(self.tokens[onset: offset])
            terminating_nodes = self.model.graph[surface_representation]
            return terminating_nodes
        else:
            # return nodes of all possible sizes
            terminating_nodes = set()
            for size in xrange(1, self.model.max_node_size + 1):
                this_size_nodes = self.get_terminating_nodes(position, size=size)
                terminating_nodes = terminating_nodes.union(this_size_nodes)
            return terminating_nodes


    def get_previous_nodes(self, node):
        """Returns"""
        pos = self.span(node)[0] -1
        return self.get_terminating_nodes(pos)

    def span(self, node):
        """Returns (tuple): indices of first and last tokens in node"""
        node_string = ' '.join(node.leaves)
        parse_string = ' '.join(self.leaves)
        try:
            pre = parse_string.split(node_string)[0]
        except:
            raise IndexError('Node({}) does not occur in parse'
                             .format(node_string))
        first = pre.count(' ')
        last = first + node_string.count(' ')
        return (first, last)


class Node(object):
    """A node in a Higraph"""
    def __init__(self, chunk_or_token):
        self.id = graph.next_id
        self.graph.weights[self.id] = 1
        self.similarity = {}

    @property
    def weight(self):
        return self.graph.weights[self.id]

    def bump_weight(self):
        self.graph.weights[self.id] += 1

    def __hash__(self):
        return self.id


if __name__ == '__main__':
    model = Model(4)
    model.read_file('small_example.txt')

    nodes = model.graph.values()
    print nodes
    exit()
    nodes = sorted(nodes, key=lambda node: node.weight)
    print nodes[:10]

