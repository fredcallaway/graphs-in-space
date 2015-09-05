from __future__ import division
import copy
import numpy as np
from collections import OrderedDict
import time

class Model(object):
    """A model that learns hierarchical chunk-based representations of sequences

    Parameters:
      max_node_size (int): the maximum number of words that can be in a chunk.
        This shouldn't exist, but is necessary for my node finding function.
    """
    def __init__(self):
        super(Model, self).__init__()
        self.max_chunk_size = 2
        self.minimum_occurences = 10  # should this be actual occurences as opposed to collocations?
        self.node_prior = 0.3
        self.num_parses = 50
        self.ftp_preference = 1
        self.graph = Higraph(self)
        self.parse_set = ParseSet(self)

    def read(self, token):
        """Reads one token and performs all associated computation"""
        # print '-'
        self.parse_set = self.parse_set.shift(token)
        self.parse_set.prune()
        self.parse_set.update_temporal_edges()
        self.graph.decay()

    def read_file(self, file_path, split=' '):
        with open(file_path) as f:
            for line in f:
                start = time.time()
                if split:
                    tokens = line.split(split)
                else:
                    tokens = list(line)  # split by charcater
                del tokens[-1]  # remove trailing \n
                for t in tokens:
                    self.read(t)
                print 'read %s tokens in %s seconds' % (len(tokens), time.time() - start)


class Higraph(OrderedDict):
    """Dict from surface representations to tokens and chunk sets"""
    def __init__(self, model):
        super(Higraph, self).__init__()
        self.model = model
        self.minimum_occurences = model.minimum_occurences
        self.node_prior = model.node_prior
        self.ftp_preference = model.ftp_preference
        # self.collocation_matrix[0][1] is count of node 1 following node 0
        self.collocation_matrix = np.zeros((100000, 10000)) # TODO: make this dynamically expand
        self.weights = np.zeros(100000)

        self._next_id = 0

    def get_chunk_id(self, chunk):
        """Returns the id of a given chunk"""
        if len(chunk) < 2:
            # singleton chunks are useless, and thus not allowed in the higraph
            return None
        similar_chunks = self[chunk.surface]  # chunks with same surface string
        for similar_chunk in similar_chunks:
            if similar_chunk == chunk:
                return similar_chunk.id
        return None

    def nodes(self):
        """Yields all nodes in the Highgraph"""
        for node_set in self.itervalues():
            for node in node_set:
                yield node

    def add(self, chunk):
        chunk.id = self.next_id
        # print 'NEW_CHUNK: ' + str(chunk)
        surface_string = chunk.surface
        if surface_string in self:
            # there is an existing node with the same surface string
            # so we add the new node to the set
            self[surface_string].add(chunk)
        else:
            self[surface_string] = {chunk}

    def bump_temporal_edge(self, pre_node, post_node):
        """Increases collocation count between pre_node and post_node by one"""
        pre_id, post_id = pre_node.id, post_node.id
        self.collocation_matrix[pre_id][post_id] += 1
        new_chunk = Chunk([pre_node, post_node], self)
        # SPEED: check conditions in an optimal order
        if (new_chunk not in self
            and np.sum(self.collocation_matrix[pre_node.id]) > self.minimum_occurences
            and np.sum(self.collocation_matrix[:, post_node.id]) > self.minimum_occurences
            and self.chunkability(pre_node, post_node) > self.node_prior):
                self.add(new_chunk)


    def ftp(self, pre_node, post_node):
        """Probability of post_node following pre_node"""
        pre_id, post_id = pre_node.id, post_node.id
        this_node_collocations = self.collocation_matrix[pre_id, post_id]
        assert isinstance(this_node_collocations, float)
        all_collocations = np.sum(self.collocation_matrix[pre_id])
        if all_collocations == 0:
            return all_collocations
        return this_node_collocations / all_collocations

    def btp(self, pre_node, post_node):
        """Probability of pre_node having preceded post_node"""
        pre_id, post_id = pre_node.id, post_node.id
        this_node_collocations = self.collocation_matrix[pre_id, post_id]
        assert isinstance(this_node_collocations, float)
        all_collocations = np.sum(self.collocation_matrix[:, post_id])
        return this_node_collocations / all_collocations

    def chunkability(self, pre_node, post_node):
        """Probability of two nodes being chunked"""
        ftp = self.ftp(pre_node, post_node)
        btp = self.btp(pre_node, post_node)
        # ftp will be considered ftp_preference times as much as btp
        weight = self.ftp_preference ** 0.5
        chunkability = (ftp * weight) * (btp / weight)
        # print 'chunkability of %s and %s: %s ' % (pre_node, post_node, chunkability)
        return chunkability

    def decay(self):
        """All weights decay"""
        pass

    @property
    def next_id(self):
        self._next_id += 1
        return self._next_id

    def __missing__(self, key):
        # add base token graph if not yet encountered
        if type(key) is int:
            # get node by id
            return self.nodes()[key]  # remember it's an OrderedDict
        elif isinstance(key, str):
            if ' ' not in key:
                # key is a base token, so we add it to the graph
                self[key] = Token(key, self)
                return self[key]
            else:
                # there are no Nodes with that surface representation
                return set()

    def __contains__(self, arg):
        if isinstance(arg, str):
            return super(Higraph, self).__contains__(arg)
        elif isinstance(arg, Chunk):
            if len(arg) == 1:
                # a chunk must have at least two elements
                return False
            nodes = self[arg.surface]
            # TODO test this
            return any([arg == node for node in nodes])
        else:
            raise ValueError()

    def __str__(self):
        nodes = ['%s:  %s' % (key, val) for key, val in self.iteritems()]
        return ('Higraph with %s nodes:\n' % len(nodes) +
                '\n'.join(nodes))


class Token(object):
    """The atomic unit."""repr__(self):
        return self.surface


class Chunk(list):
    """The basic grammatical unit: a list of Chunks and Tokens.

    All subchunks must be in the Higraph?
    """
    def __init__(self, new_chunk, graph):
        """
        Args:
          new_chunk: a list of Chunks, or a single string representing a base token
        """
        super(Chunk, self).__init__()
        self.graph = graph
        self.leaves = []

        first_sub_chunk = new_chunk[0]
        self.append(first_sub_chunk)
        self.leaves.extend(first_sub_chunk.leaves)
        self.probability = 1  # get the first word for free

        if len(new_chunk) > 1:
            for chunk in new_chunk[1:]:
                self.shift(chunk)
        
        self.id = self.graph.get_chunk_id(self)

    def __hash__(self):
        if self.id is not None:
            return self.id
        else:
            raise ValueError('Cannot hash a chunk not in the higraph')

    @property
    def surface(self):
        return ' '.join(self.leaves)


    def __repr__(self):
        if len(self) > 1:
            return '[' + ' '.join(map(str, self)) + ']'
        else:
            return str(self[0])

    def __eq__(self, other):
        if self.surface == other.surface:  # check this first for efficiency
            # recursively check that subchunks are equal
            # note that computation stops as soon as a comparison returns False
            return all((self_chunk == other_chunk
                        for self_chunk, other_chunk in zip(self, other)))

class Parse(list):
    """A list of Chunks with a probability"""
    def __init__(self):
        super(Parse, self).__init__()
        self.probability = 1

    def shift(self, chunk_or_token):
        """Adds a chunk or token to this chunk and adjusts probability"""
        # TODO if we make a new chunk, we should try to merge with previous chunk
        self.append(chunk_or_token)
        self.leaves.extend(chunk_or_token.leaves)  # SPEED: do we need to keep track of leaves?
        # should probability include btp like chunkability?
        self.probability *= self.graph.ftp(self[-2], self[-1])  # markov assumption

    def unshift(self, num_chunks):
        """Removes the specified number of chunks from tail and adjusts probability.

        Inverse of shift."""
        for _ in xrange(num_chunks):
            self.probability /= self.graph.ftp(self[-2], self[-1])
            num_leaves = len(self[-1].leaves)
            del self.leaves[-num_leaves:]
            del self[-1]

class ParseSet(list):
    """A set of possible chunkings for a surface string.

    This constitutes the model's representation of the input that is currently
    in memory"""
    def __init__(self, model):
        super(ParseSet, self).__init__()
        self.model = model

    def shift(self, token):
        """Returns a new new ParseSet with `token` appended to the surface string"""
        token = self.model.graph[token]  # convert str to Token
        new_parse_set = ParseSet(self.model)
        # a "chunking" is just a chunk, but one that represents the utterance of this ParseSet
        if not self:  # ParseSet is empty
            new_parse_set.append(Parse([token], self.model.graph))
        for old_parse in self:
            simple_parse = copy.copy(old_parse)
            # in the simplest case, we append without chunking anything
            simple_parse.shift(token)
            new_parse_set.append(simple_parse)

            # for each node that this token completes, add a possible parse
            for chunk_size in range(2, self.model.max_node_size + 1):
                possible_chunk = Chunk(simple_parse[-chunk_size:], self.model.graph)
                if possible_chunk.id is not None:
                    parse = copy.copy(simple_parse)
                    parse.unshift(chunk_size)  # remove chunks we are replacing
                    parse.shift(possible_chunk)
                    new_parse_set.append(parse)

        return new_parse_set

    def update_temporal_edges(self):
        """Bumps collocation count for the two most recent chunks in each chunking"""
        for parse in self:
            try:
                self.model.graph.bump_temporal_edge(parse[-2], parse_set[-1])
            except IndexError:
                pass

    def prune(self):
        """Removes least likely parses"""
        self.sort(key=lambda chunk: chunk.probability)
        del self[:-self.model.num_parses]


# def string_to_chunk(string):
#     '[this [big dog]]'

#     '[I [am [going to] [the store]]]'
#     ['this' ['big', 'dog']]

#     for element in list_chunk:
#         if isinstance(element, str):
#             Token


if __name__ == '__main__':

    start = time.time()
    m = Model()
    m.read_file('letters.txt', None)
    print m.graph
    print 'RUNTIME: %s' % (time.time() - start)
    
                
