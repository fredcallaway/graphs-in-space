from collections import Counter, OrderedDict, deque
import itertools
from typing import Dict, List
import numpy as np
import yaml
import utils

from holograph import HoloGraph, Node


LOG = utils.get_logger(__name__, stream='INFO', file='WARNING')


class Numila(object):
    """The premier language acquisition model."""
    def __init__(self, param_file='params.yml', **parameters) -> None:
        # read parameters from file, overwriting with keyword arguments
        with open(param_file) as f:
            self.params = yaml.load(f.read())
        self.params.update(parameters)
        self.graph = HoloGraph(edges=['ftp', 'btp'], params=self.params)

    def parse_utterance(self, utterance, learn=True, verbose=False) -> 'Parse':
        self.graph.decay()
        if isinstance(utterance, str):
            utterance = utterance.split(' ')
        utterance = ['#'] + utterance + ['#']
        return Parse(self, utterance, learn=learn, verbose=verbose)

    def create_chunk(self, node1, node2) -> Node:
        """Returns a chunk composed of node1 and node2.

        This just creates a node. It does not add it to the graph.
        """
        # A chunk's id_string is determined by its constituents id_strings,
        # allowing us to quickly find a chunk given its constituents.
        chunk_id_string = '[{node1.id_string} {node2.id_string}]'.format_map(locals())
        
        # We break violate the modularity of HoloGraph here in order
        # to get compositional structure in the id_vectors.
        id_vec = self.graph.vector_model.bind(node1.id_vec, node2.id_vec)
        return self.graph.create_node(chunk_id_string, id_vec)

    def get_chunk(self, node1, node2, force=False) -> Node:
        """Returns a chunk of node1 and node2 if the chunk is in the graph.

        If the chunk doesn't exist, we check if the pair is chunkable
        enough for a new chunk to be created. If so, the new chunk is returned.
        """
        # See note regarding chunk id string in `create_chunk()`.
        chunk_id_string = '[{node1.id_string} {node2.id_string}]'.format_map(locals())
        if chunk_id_string in self.graph:
            return self.graph[chunk_id_string]
        else:
            # consider making a new node
            if self.chunkability(node1, node2) > self.params['CHUNK_THRESHOLD']:
                chunk = self.create_chunk(node1, node2)
                self.graph.add_node(chunk)
                return chunk
            else:
                if force:
                    return self.create_chunk(node1, node2)
                else:
                    return None

    def chunkability(self, node1, node2, generalize=0.0) -> float:
        """How well two nodes form a chunk.

        The geometric mean of forward transitional probability and
        bakward transitional probability."""

        ftp = self.graph.edge_weight('ftp', node1, node2, 
                                     generalize=self.params['GENERALIZE'])
        btp = self.graph.edge_weight('btp', node2, node1, 
                                     generalize=self.params['GENERALIZE'])
        if ftp < 0 or btp < 0:
            return 0.0
        else:
            return (ftp * btp) ** 0.5


    def speak(self, words, verbose=False) -> Node:
        log = print if verbose else lambda *args: None
        nodes = [self.graph.safe_get(w) for w in words]
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
            log('\tchunk:', chunk)
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
        self.log = print if verbose else lambda *args: None  # a dummy function
        self.log('\nPARSING', '"', ' '.join(utterance), '"')

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

        self.log('no more tokens')
        # Process the tail end.
        while self.memory:  # there are nodes left to be processed
            self.update_weights()
            self.try_to_chunk()


    def shift(self, token) -> None:
        """Adds a token to memory, creating a new node in the graph if new.

        If 4 or more nodes are already in memory, shifting a new token will
        implicitly drop the least recent node from memory.
        """
        assert len(self.memory) < self.memory.maxlen

        self.log('shift: {token}'.format_map(locals()))
        try:
            node = self.graph[token]
        except KeyError:  # a new token
            node = self.graph.create_node(token)
            if self.learn:
                self.graph.add_node(node)
        self.memory.append(node)

    def update_weights(self) -> None:
        """Strengthens the connection between every adjacent pair of Nodes in memory.

        For a pair (the, dog) we increase the weight of  the forward edge,
        the -> dog, and the backward edge, dog -> the."""
        if not self.learn:
            return
        self.log('memory =', self.memory)

        ftp_factor = self.params['LEARNING_RATE'] * self.params['FTP_PREFERENCE'] 
        btp_factor = self.params['LEARNING_RATE'] * (1 - self.params['FTP_PREFERENCE']) 

        for node, next_node in utils.neighbors(self.memory):
            self.log('  -> strengthen: {node} & {next_node}'.format_map(locals()))
            self.graph.bump_edge('ftp', node, next_node, ftp_factor)
            self.graph.bump_edge('btp', next_node, node, btp_factor)
            

    def try_to_chunk(self) -> None:
        """Attempts to combine two Nodes in memory into one Node.

        If no Node pairs form a chunk, then the oldest node in memory is
        dropped from memory. Thus, this method always reduces the numbe of
        nodes in memory by 1."""
        #memory = self[-memory_size:]

        if len(self.memory) == 1:
            # We can't create a chunk when there's only one node left.
            # This can only happen while processing the tail, so we
            # must be done processing
            self.append(self.memory.popleft())
            return
        
        chunkabilities = [self.model.chunkability(node, next_node)
                          for node, next_node in utils.neighbors(self.memory)]
        self.log('chunkabilities =', chunkabilities)
        best_idx = np.argmax(chunkabilities)

        chunk = self.model.get_chunk(self.memory[best_idx], self.memory[best_idx+1])
        if chunk:
            # combine the two nodes into one chunk
            self.log(('  -> create chunk: {chunk}').format_map(locals()))
            self.memory[best_idx] = chunk
            del self.memory[best_idx+1]
        else:  # can't make a chunk
            # We remove the oldest node to make room for a new one.
            self.append(self.memory.popleft())
            self.log('  -> no chunk created')

    def __str__(self):
        string = super().__str__().replace(',', ' |')[1:-1]
        return '(' + string + ')'
