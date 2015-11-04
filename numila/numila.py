from collections import Counter, OrderedDict
from typing import Dict, List
from typed import typechecked
import numpy as np
import yaml
import utils
import vectors


LOG = utils.get_logger(__name__, stream='INFO', file='WARNING')


class Node(object):
    """A Node in a graph.

    Attributes:
        string: e.g. [the [big dog]]
        idx: an int identifier
        forward_edges: the number of times each other node in the graph has been
                 after this node in the memory window
        backward_edges: the number of times each other node in the graph has been
                 before this node in the memory window
        id_vector: a random sparse vector that never changes
    """
    def __init__(self, graph, string, idx, id_vector) -> None:
        self.graph = graph
        self.params = graph.params
        self.string = string
        self.idx = idx

        self.id_vector = id_vector
        self.semantic_vec = np.copy(self.id_vector)
        
        self.count = 0
        self.forward_edges = Counter()
        self.backward_edges = Counter()

    @property
    def context_vec(self) -> np.ndarray:
        """Represents this node in context."""
        return (self.semantic_vec * self.params['SEMANTIC_TRANSFER'] +
                self.id_vector * (1 - self.params['SEMANTIC_TRANSFER']))

    @property
    def precede_vec(self) -> np.ndarray:
        """Represents this node occurring before another node"""
        return np.roll(self.context_vec, 1)

    @property
    def follow_vec(self) -> np.ndarray:
        """Represents this node occurring after another node"""
        return np.roll(self.context_vec, -1)  # TODO: parameterize this

    def decay(self) -> None:
        """Makes learned vector more similar to initial random vector."""
        self.semantic_vec += self.id_vector * self.params['DECAY_RATE']

    def distribution(self, kind, use_vectors=True, exp=8):
        if use_vectors:
            if kind is 'following':
                dists = [vectors.cosine(self.semantic_vec, node2.follow_vec)
                         for node2 in self.graph.nodes]
            elif kind is 'preceding':
                dists = [vectors.cosine(self.semantic_vec, node2.precede_vec)
                         for node2 in self.graph.nodes]
            elif kind is 'chunking':
                dists = [self.graph.chunkability(self, node2) for node2 in self.graph.nodes]
            else:
                raise ValueError('{kind} is not a valid distribution.'
                                 .format_map(locals()))

            distribution = (np.array(dists) + 1.0) / 2.0  # probabilites must be non-negative
            distribution **= exp  # accentuate differences
            return distribution / np.sum(distribution)

        else:  # use counts
            if kind is 'following':
                edge = 'forward_edges'
            elif kind is 'preceding':
                edge = 'backward_edges'
            else:
                raise ValueError('{kind} is not a valid distribution.'
                                 .format_map(locals()))

            counts = np.array([getattr(self, edge)[node2]
                               for node2 in self.graph.nodes])
            total = np.sum(counts)
            if total == 0:
                # This can happen for a node that never ends/begins an 
                # utterance. For example, '[ate #]' never begins an utterance.
                return counts
            else:
                return counts / np.sum(counts)

    def predict(self, kind, **distribution_args):
        """Returns the node most likely to follow this node."""
        distribution = self.distribution(kind, **distribution_args)
        return np.random.choice(self.graph.nodes, p=distribution)

    def __hash__(self) -> int:
        return self.idx

    def __repr__(self):
        return self.string

    def __str__(self):
        return self.string



class Parse(list):
    """A parse of an utterance represented as a list of Nodes. 

    The parse is computed upon intialization. This computation has side
    effects for the parent Numila instance (i.e. learning)."""
    def __init__(self, graph, utterance, verbose=False) -> None:
        super(Parse, self).__init__()
        self.graph = graph
        self.params = graph.params
        self.log = print if verbose else lambda *args: None  # a dummy function
        self.log('\nPARSING', '"', ' '.join(utterance), '"')

        for token in utterance:
            self.graph.decay()  # model's knowledge decays on every word it hears
            self.shift(token)
            self.update_weights(self.params['MEMORY_SIZE'])
            if len(self) >= self.params['MEMORY_SIZE']:
                # fill memory before trying to chunk
                # possibly replace two nodes in working memory with one node
                self.try_to_chunk(self.params['MEMORY_SIZE'])

        # Process the tail end. We have to shrink memory_size to prevent
        # accessing elements that fell out of the 4 item memory window.
        self.log('no more tokens')
        for memory_size in range(self.params['MEMORY_SIZE']-1, 1, -1):
            self.update_weights(memory_size)
            self.try_to_chunk(memory_size)


    def shift(self, token) -> None:
        self.log('shift: {token}'.format_map(locals()))
        try:
            node = self.graph[token]
        except KeyError:  # a new token
            node = self.graph.create_token(token)
        node.count += 1
        self.append(node)

    def update_weights(self, memory_size) -> None:
        """Strengthens the connection between every adjacent pair of Nodes in memory."""
        memory = self[-memory_size:]
        self.log('memory =', memory)

        # We have to make things a little more complicated to avoid
        # updating based on vectors changed in this round of updates.
        factor = self.params['LEARNING_RATE'] * self.params['FTP_PREFERENCE'] 
        ftp_updates = {node: (node.follow_vec * factor)
                       for node in memory[1:]}

        factor = self.params['LEARNING_RATE'] * (1 - self.params['FTP_PREFERENCE']) 
        btp_updates = {node: (node.precede_vec * factor)
                       for node in memory[:-1]}

        for i in range(len(memory) - 1):
            node = memory[i]
            next_node = memory[i + 1]
            self.log('  -> strengthen: {node} & {next_node}'.format_map(locals()))

            node.forward_edges[next_node] += 1
            next_node.backward_edges[node] += 1
            
            node.semantic_vec += ftp_updates[next_node]
            next_node.semantic_vec += btp_updates[node]

    def try_to_chunk(self, memory_size) -> None:
        """Attempts to combine two Nodes in memory into one Node.

        If no Node pairs form a chunk, then nothing happens. """
        memory = self[-memory_size:]
        if len(memory) < 2:
            return  # can't chunk with less than 2 nodes
        
        # invariant: idx is the index of the earlier Node under consideration
        chunkabilities = []
        for idx in range(len(memory) - 1):
            node = memory[idx]
            next_node = memory[idx+1]

            chunkability = self.graph.chunkability(node, next_node)
            chunkabilities.append(chunkability)
        self.log('chunkabilities =', chunkabilities)
    
        best_idx = chunkabilities.index(max(chunkabilities))
        parse_idx = best_idx - len(memory)  # convert index in memory to index in parse
        assert parse_idx < -1  # must be an index from tail, and not the last element

        chunk = self.graph.get_chunk(self[parse_idx], self[parse_idx+1])
        if chunk:
            self.log(('  -> create chunk: {chunk}').format_map(locals()))
            # combine the two nodes into one chunk
            chunk.count += 1
            self[parse_idx] = chunk
            del self[parse_idx+1]
        else:
            self.log('  -> no chunk created')

    def __str__(self):
        return super(Parse, self).__str__().replace(',', '')


class Numila(object):
    """The premier language acquisition model"""
    def __init__(self, param_file='params.yml', **parameters) -> None:
        super(Numila, self,).__init__()

        # read parameters from file, overwriting with keyword arguments
        with open(param_file) as f:
            self.params = yaml.load(f.read())
        self.params.update(parameters)
        self.vector_model = vectors.VectorModel(self.params['DIM'],
                                                self.params['PERCENT_NON_ZERO'])
        
        # Each token gets an int ID which specifies its index
        # in self.nodes and self.activations.
        self.string_to_index = OrderedDict()  # type: Dict[str, int]
        self.nodes = []
        self.activations = np.zeros(100000)

    def parse_utterance(self, utterance, verbose=False):
        if isinstance(utterance, str):
            utterance = utterance.split()
        utterance = ['#'] + utterance + ['#']
        return Parse(self, utterance, verbose)

    def create_token(self, token_string) -> Node:
        """Add a new base token to the graph."""
        id_vector = self.vector_model.sparse()
        return self._create_node(token_string, id_vector)

    def create_chunk(self, node1, node2) -> Node:
        """Add a new chunk to the graph composed of node1 and node2."""
        chunk_string = '[{node1.string} {node2.string}]'.format_map(locals())
        id_vector = self.vector_model.bind(node1.context_vec, node2.context_vec)
        return self._create_node(chunk_string, id_vector)

    def _create_node(self, node_string, id_vector) -> Node:
        idx = len(self.nodes)
        new_node = Node(self, node_string, idx, id_vector)
        self.string_to_index[node_string] = idx
        self.nodes.append(new_node)
        self.activations[idx] = 1.0
        return new_node

    def get_chunk(self, node1, node2) -> Node:
        """Returns a chunk of node1 and node2 if the chunk is in the graph.

        If the chunk doesn't exist, we check if the pair is chunkable
        enough for a new chunk to be created. If so, the new chunk is returned.
        """
        chunk_string = '[{node1.string} {node2.string}]'.format_map(locals())
        if chunk_string in self:
            return self[chunk_string]
        else:
            # consider making a new node
            if self.chunkability(node1, node2) > self.params['CHUNK_THRESHOLD']:
                return self.create_chunk(node1, node2)
            else:
                return None

    def chunkability(self, node1, node2) -> float:
        """How well two nodes form a chunk."""

        # TODO check if the nodes already make a node?
        # ftp looks at the filler identity vector, thus is parallel to decoding of self.params['BEAGLE']
        ftp = vectors.cosine(node1.semantic_vec, node2.follow_vec)
        btp = vectors.cosine(node1.precede_vec, node2.semantic_vec)
        return (ftp + btp) / 2

    def decay(self) -> None:
        for node in self.nodes:
            node.decay()

    def speak(self, **distribution_args):
        utterance = [self['#']]
        for _ in range(20):
            nxt = utterance[-1].predict('following', **distribution_args)
            utterance.append(nxt)
            if '#' in str(nxt):  # could be a chunk with boundary char
                break
        return ' '.join(map(str, utterance))

    def __getitem__(self, node_string) -> Node:
        #if node_string in self.string_to_index:
        try:
            idx = self.string_to_index[node_string]
            return self.nodes[idx]
        except KeyError:
            raise KeyError('{node_string} is not in the graph.'.format_map(locals()))
            
        #else:
        #    if ' ' in node_string:
        #        raise ValueError('{node_string} is not in the graph.'.format_map(locals()))
        #    # node_string represents a new token, so we make a new Node for it
        #    return self.create_token(node_string)

    def __contains__(self, item) -> bool:
        assert isinstance(item, str)
        return item in self.string_to_index
