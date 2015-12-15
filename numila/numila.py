from collections import Counter, OrderedDict, deque
import itertools
from typing import Dict, List
import numpy as np
import yaml
import utils

import holograph

LOG = utils.get_logger(__name__, stream='INFO', file='WARNING')


class Node(holograph.HoloNode):
    """A node in Numila's graph."""
    def __init__(self, graph, id_string, id_vec=None):
        super().__init__(graph, id_string, id_vec)
        self.followers= set()  # TODO document
        self.predecessors = set()
        

class Chunk(Node):
    """A chunk of two nodes."""
    def __init__(self, graph, node1, node2):
        self.child1 = node1
        self.child2 = node2

        # A chunk's id_string is determined by its constituents id_strings,
        # allowing us to quickly find a chunk given its constituents.
        chunk_id_string = '[{node1.id_string} {node2.id_string}]'.format_map(locals())
        
        # We break violate the modularity of HoloGraph here in order
        # to get compositional structure in the id_vectors.
        id_vec = graph.vector_model.bind(node1.id_vec, node2.id_vec)
        
        super().__init__(graph, chunk_id_string, id_vec=id_vec)

class Numila(object):
    """The premier language acquisition model."""
    def __init__(self, param_file='params.yml', **parameters) -> None:
        # read parameters from file, overwriting with keyword arguments
        with open(param_file) as f:
            self.params = yaml.load(f.read())
        self.params.update(parameters)
        self.graph = holograph.HoloGraph(edges=['ftp', 'btp'], params=self.params)

    def parse_utterance(self, utterance, learn=True, verbose=False) -> 'Parse':
        self.graph.decay()
        if isinstance(utterance, str):
            utterance = utterance.split(' ')
        utterance = ['#'] + utterance + ['#']
        return Parse(self, utterance, learn=learn, verbose=verbose)

    def get_chunk(self, node1, node2, try_create=False, force=False) -> Node:
        """Returns a chunk of node1 and node2 if the chunk is in the graph.

        If the chunk doesn't exist, we check if the pair is chunkable
        enough for a new chunk to be created. If so, the new chunk is returned.
        """
        # See note regarding chunk id string in `Chunk.__init__()`
        chunk_id_string = '[{node1.id_string} {node2.id_string}]'.format_map(locals())
        if chunk_id_string in self.graph:
            return self.graph[chunk_id_string]
        if try_create:
            # consider making a new chunk
            if self.chunkability(node1, node2) > self.params['CHUNK_THRESHOLD']:
                chunk = Chunk(self.graph, node1, node2)
                self.graph.add_node(chunk)
                node1.followers.add(node2)
                node2.predecessors.add(node1)
                LOG.debug('new chunk: %s', chunk)
                return chunk
        if force:
            return Chunk(self.graph, node1, node2)


    def chunkability(self, node1, node2, generalize=None) -> float:
        """How well two nodes form a chunk.

        The geometric mean of forward transitional probability and
        bakward transitional probability.
        """
        assert node1; assert node2

        if generalize is None:
            generalize = self.params['GENERALIZE']
        if generalize:
            similar_chunks = (self.get_chunk(predecessor, follower)
                              for predecessor in node2.predecessors
                              for follower in node1.followers)
            similar_chunks = [c for c in similar_chunks if c is not None]
            LOG.debug(str(similar_chunks))

            gen_chunkability =  sum(self.chunkability(chunk.child1, chunk.child2, 
                                                      generalize=0)
                                    for chunk in similar_chunks)
            return (generalize * gen_chunkability + 
                    (1-generalize) * self.chunkability(node1, node2, generalize=0))
        else:
            ftp = node1.edge_weight('ftp', node2)
            btp = node2.edge_weight('btp', node1)
            if ftp < 0 or btp < 0:
                # chunkability can't be below 0
                return 0.0
            else:
                return (ftp * btp) ** 0.5  # geometric mean

    def speak(self, words, verbose=False) -> Node:
        """Returns a single node containing all of `words`."""
        def get_node(token):
            try:
                return self.graph[token]
            except KeyError:
                return Node(self.graph, token)
        nodes = [get_node(w) for w in words]
        np.random.shuffle(nodes)

        # combine the two chunkiest nodes into a chunk until only one node left
        while len(nodes) > 1:
            node1, node2 = max(itertools.permutations(nodes, 2),
                         key=lambda pair: self.chunkability(*pair))
            nodes.remove(node1)
            nodes.remove(node2)
            # Using `force=True` returns the best chunk, regardless of 
            # whether it is a  chunk in the graph or not.
            chunk = self.get_chunk(node1, node2, force=True)
            assert chunk
            LOG.debug('\tchunk: %s', chunk)
            nodes.append(chunk)

        return nodes[0]

class Parse(list):
    """A parse of an utterance represented as a list of Nodes. 

    The parse is computed upon intialization. This computation has side
    effects for the parent Numila instance (i.e. learning). The loop
    in __init__ is thus the learning algorithm.
    """
    def __init__(self, model, utterance, learn=True, verbose=False) -> None:
        super().__init__()
        self.model = model
        self.graph = model.graph
        self.params = model.params
        self.learn = learn
        self.memory = deque(maxlen=self.params['MEMORY_SIZE'])
        LOG.debug('\nPARSING: %s', ' '.join(utterance))

        # This is the "main loop" of the model, i.e. it will run once for
        # every token in the training corpus. The four functions in the loop
        # correspond to the four steps of the learning algorithm.
        for token in utterance:
            #self.graph.decay()
            self.shift(token)
            self.update_weights()
            if len(self.memory) == self.params['MEMORY_SIZE']:
                # Only attempt to chunk after filling memory.
                self.try_to_chunk()  # always decreases number of nodes in memory by 1

        LOG.debug('no more tokens')
        # Process the tail end.
        while self.memory:  # there are nodes left to be processed
            self.update_weights()
            self.try_to_chunk()


    def shift(self, token) -> None:
        """Adds a token to memory.

        This method can only be called when there is an empty slot in
        memory. If token has never been seen, this method creates a new
        node in the graph for it.
        """
        assert len(self.memory) < self.memory.maxlen
        LOG.debug('shift: %s', token)

        try:
            node = self.graph[token]
        except KeyError:  # a new token
            node = Node(self.graph, token)
            if self.learn:
                self.graph.add_node(node)

        self.memory.append(node)
        LOG.debug('memory = %s', self.memory)

    def update_weights(self) -> None:
        """Strengthens the connection between every adjacent pair of nodes in memory.

        For a pair (the, dog) we increase the weight of  the forward edge,
        the -> dog, and the backward edge, dog -> the."""
        if not self.learn:
            return

        ftp_factor = self.params['LEARNING_RATE'] * self.params['FTP_PREFERENCE'] 
        btp_factor = self.params['LEARNING_RATE'] * (1 - self.params['FTP_PREFERENCE']) 

        for node1, node2 in utils.neighbors(self.memory):
            LOG.debug('  -> strengthen: %s & %s', node1, node2)
            node1.bump_edge('ftp', node2, ftp_factor)
            node2.bump_edge('btp', node1, btp_factor)

    def try_to_chunk(self) -> None:
        """Attempts to combine two Nodes in memory into one Node.

        If no Node pairs form a chunk, then the oldest node in memory is
        dropped from memory. Thus, this method always reduces the numbe of
        nodes in memory by 1.
        """

        if len(self.memory) == 1:
            # We can't create a chunk when there's only one node left.
            # This can only happen while processing the tail, so we
            # must be done processing
            self.append(self.memory.popleft())
            return
        
        # Find the pair of adjacent nodes with highest chunkability.
        chunkabilities = [self.model.chunkability(node, next_node)
                          for node, next_node in utils.neighbors(self.memory)]
        LOG.debug('chunkabilities = ' + str(chunkabilities))
        best_idx = np.argmax(chunkabilities)
        chunk = self.model.get_chunk(self.memory[best_idx], 
                                     self.memory[best_idx+1],
                                     try_create=True)

        if chunk:
            # combine the two nodes into one chunk
            LOG.debug('  -> create chunk: {chunk}'.format_map(locals()))
            self.memory[best_idx] = chunk
            del self.memory[best_idx+1]
        else:  # can't make a chunk
            # We remove the oldest node to make room for a new one.
            self.append(self.memory.popleft())
            LOG.debug('  -> no chunk created')

    def __str__(self):
        string = super().__str__().replace(',', ' |')[1:-1]
        return '(' + string + ')'
