from collections import deque
import itertools
import numpy as np
import yaml
import utils


LOG = utils.get_logger(__name__, stream='INFO', file='INFO')
TEST = {'on': False}


class Numila(object):
    """The premier language acquisition model."""
    def __init__(self, param_file='params.yml', **params) -> None:
        LOG.info('parameters: %s', params)

        # Read default params from file, overwriting with keyword arguments.
        with open(param_file) as f:
            self.params = yaml.load(f.read())
        assert(all(k in self.params for k in params))
        self.params.update(params)

        # The GRAPH parameter determines which implementation of a Graph
        # this Numila instance should use. Thus, Numila is a class that
        # is parameterized by another class, similarly to how a functor
        # in OCaml is a module parameterized by another module.
        graph = self.params['GRAPH'].lower()
        if graph == 'holograph':
            from holograph import HoloGraph
            Graph = HoloGraph
        elif graph == 'probgraph':
            from probgraph import ProbGraph
            Graph = ProbGraph
        else:
            raise ValueError('Invalid GRAPH parameter: {}'.format(self.params['GRAPH']))
        
        self.graph = Graph(edges=['ftp', 'btp'], params=self.params)

        # This is kind of crazy. Each Numila instance has its own Node and
        # Chunk classes. These two classes both inherit from the Node class
        # of the graph. We must use simple helper functions to allow for
        # this kind of dynamic inheritance.
        self.Node = make_node_class(Graph.Node)
        self.Chunk = make_chunk_class(self.Node)


    def parse_utterance(self, utterance, learn=True, verbose=False):
        """Returns a Parse of the given utterance."""
        self.graph.decay()
        if isinstance(utterance, str):
            utterance = utterance.split(' ')
        if self.params['ADD_BOUNDARIES']:
            utterance = ['#'] + utterance + ['#']
        return Parse(self, utterance, learn=learn, verbose=verbose)

    def fit(self, training_corpus):
        with utils.Timer('Numila train time'):
            for utt in training_corpus:
                self.parse_utterance(utt)
            return self

    def get_chunk(self, node1, node2, stored_only=True):
        """Returns a chunk of node1 and node2 if the chunk is in the graph.

        If stored_only is True, we only return the desired chunk if it
        has been stored as an exemplar in the graph. Otherwise, we
        always return a chunk, creating it if necessary.

        If the chunk doesn't exist, we check if the pair is chunkable
        enough for a new chunk to be created. If so, the new chunk is returned.
        """
        # See note regarding chunk id string in `Chunk.__init__()`
        chunk_id_string = '[{node1.id_string} {node2.id_string}]'.format_map(locals())
        if chunk_id_string in self.graph:
            return self.graph[chunk_id_string]
            
        if not stored_only:
            if node1.id_string in self.graph and node1 is not self.graph[node1.id_string]:
                LOG.warning('Fixing a chunk node')
                node1 = self.graph[node1.id_string]
            if node2.id_string in self.graph and node2 is not self.graph[node2.id_string]:
                LOG.warning('Fixing a chunk node')
                node2 = self.graph[node2.id_string]
            chunk = self.Chunk(self, node1, node2)
            return chunk

    def add_chunk(self, chunk):
        if (chunk.child1.id_string not in self.graph or
            chunk.child2.id_string not in self.graph):
                # This is a strange edge case that can happen when there is
                # a low exemplar threshold. We just move on without adding
                # the chunk.
                LOG.info('Tried to add a chunk with a non-chunk child: %s', chunk)
                return
        self.graph.add_node(chunk)
        assert chunk.child1 is self.graph[chunk.child1.id_string]
        assert chunk.child2 is self.graph[chunk.child2.id_string]
        chunk.child1.followers.add(chunk.child2)
        chunk.child2.predecessors.add(chunk.child1)

        LOG.debug('new chunk: %s', chunk)

    def speak(self, words, verbose=False, return_chunk=False):
        """Returns the list of words ordered properly."""
        def get_node(token):
            try:
                return self.graph[token]
            except KeyError:
                LOG.info('Unknown token seen while speaking.')
                return self.Node(self, token)
        nodes = [get_node(w) for w in words]

        # combine the two chunkiest nodes into a chunk until only one node left
        while len(nodes) > 1:
            chunks = [self.get_chunk(node1, node2, stored_only=False)
                      for node1, node2 in itertools.permutations(nodes, 2)]
            best_chunk = max(chunks, key=lambda chunk: chunk.chunkiness())
            nodes.remove(best_chunk.child1)
            nodes.remove(best_chunk.child2)
            LOG.debug('\tchunk: %s', best_chunk)
            nodes.append(best_chunk)

        final = nodes[0]
        if return_chunk:
            return final
        else:
            return utils.flatten_parse(final)


class Parse(list):
    """A parse of an utterance represented as a list of Nodes.

    The parse is computed upon intialization. This computation has side
    effects for the parent Numila instance (i.e. learning). The loop
    in __init__ is thus both the comprehension and learning algorithm.
    """
    def __init__(self, model, utterance, learn=True, verbose=False) -> None:
        super().__init__()
        self.model = model
        self.utterance = utterance
        self.graph = model.graph
        self.params = model.params
        self.learn = learn
        self.log_chunkiness = 0

        self.memory = deque(maxlen=self.params['MEMORY_SIZE'])
        self.log = print if verbose else lambda *args: None  # dummy function
        self.log('\nPARSING: %s', ' '.join(utterance))

        # This is the "main loop" of the model, i.e. it will run once for
        # every token in the training corpus.
        for token in utterance:
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
        """Adds a token to memory.

        This method can only be called when there is an empty slot in
        memory. If token has never been seen, this method creates a new
        node in the graph for it.
        """
        assert len(self.memory) < self.memory.maxlen
        self.log('shift: %s', token)

        try:
            node = self.graph[token]
        except KeyError:  # a new token
            node = self.model.Node(self.model, token)
            if self.learn:
                self.graph.add_node(node)

        self.memory.append(node)
        self.log('memory = %s', self.memory)

    def update_weights(self) -> None:
        """Strengthens the connection between every adjacent pair of nodes in memory.

        Additionally, we increase the connection between nodes that are in a 
        chunk in memory. For a pair ([the big], dog) we increase the weight of 
        the forward edge, [the big] -> dog, and the backward edge, dog -> [the big],
        as well as the forward edge, the -> big, and backward edge, big -> the.
        """
        if not self.learn:
            return

        # These factors determine how much we should increase the weight
        # of each type of edge.
        ftp_factor = self.params['LEARNING_RATE'] * self.params['FTP_PREFERENCE'] 
        btp_factor = self.params['LEARNING_RATE'] * (1 - self.params['FTP_PREFERENCE']) 

        # Increase the weight between every node in memory. Note that
        # some of these nodes may be chunks that are not in the graph. TODO
        for node1, node2 in utils.neighbors(self.memory):
            self.log('  -> strengthen: %s & %s', node1, node2)
            if node1.id_string in self.graph and node2.id_string in self.graph:
                node1.bump_edge('ftp', node2, ftp_factor)
                node2.bump_edge('btp', node1, btp_factor)

        if self.params['CHUNK_LEARNING']:
            # Incerase the weight between nodes that are in chunks in memory.
            ftp_factor *= self.params['CHUNK_LEARNING']
            btp_factor *= self.params['CHUNK_LEARNING']

            def update_chunk(chunk):
                """Recursively updates weights between elements of the chunk"""
                if not hasattr(chunk, 'chunkiness'):
                    return
                if chunk.child1.id_string in self.graph and chunk.child2.id_string in self.graph:
                    chunk.child1.bump_edge('ftp', chunk.child2, ftp_factor)
                    chunk.child2.bump_edge('btp', chunk.child1, btp_factor)
                update_chunk(chunk.child1)
                update_chunk(chunk.child2)

            for node in self.memory:
                update_chunk(node)

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

        chunks = [self.model.get_chunk(node1, node2, stored_only=False)
                  for node1, node2 in utils.neighbors(self.memory)]
        chunkinesses = [chunk.chunkiness() for chunk in chunks]
        best_idx = np.argmax(chunkinesses)
        best_chunkiness = chunkinesses[best_idx]
        best_chunk = chunks[best_idx]

        if best_chunkiness >= self.params['CHUNK_THRESHOLD']:  # TODO chunk threshold
            # Replace the two nodes in memory with one chunk.
            self.log('  -> create chunk: {}'.format(best_chunk))
            self.memory[best_idx] = best_chunk
            del self.memory[best_idx+1]
            self.log_chunkiness += np.log(best_chunkiness)

            # Add the chunk to the graph if it exceeds a threshold chunkiness.
            if (self.learn and
                best_chunk.id_string not in self.graph and
                best_chunk.chunkiness() > self.params['EXEMPLAR_THRESHOLD']):
                    self.model.add_chunk(best_chunk)      
        else:  # can't make a chunk
            # We remove the oldest node to make room for a new one.
            self.append(self.memory.popleft())
            self.log('  -> no chunk created')

    def __repr__(self):
        return super().__repr__().replace(',', ' |')[1:-1]

    def __str__(self):
        return self.__repr__()


def make_node_class(BaseNode):
    """Returns a Node class that inherits from BaseNode"""
    
    class Node(BaseNode):
        """A node in Numila's graph."""
        def __init__(self, model, id_string):
            super().__init__(model.graph, id_string)
            self.followers = set()  # TODO document
            self.predecessors = set()

    return Node


def make_chunk_class(Node):
    """Returns a Chunk class that inherits from Node."""

    class Chunk(Node):
        """A chunk of two nodes."""
        def __init__(self, model, node1, node2):
            self.model = model
            self.child1 = node1
            self.child2 = node2

            # A chunk's id_string is determined by its constituents id_strings,
            # allowing us to quickly find a chunk given its constituents.
            chunk_id_string = '[{node1.id_string} {node2.id_string}]'.format_map(locals())

            # We violate the modularity of HoloGraph here in order
            # to get compositional structure in the id_vectors.
            #id_vec = model.graph.vector_model.bind(node1.id_vec, node2.id_vec)

            super().__init__(model, chunk_id_string)

        def chunkiness(self, generalize=None) -> float:
            """How well two nodes form a chunk.

            The geometric mean of forward transitional probability and
            bakward transitional probability.
            """

            if generalize is None:
                generalize = self.model.params['GENERALIZE']

            if not generalize:
                ftp = self.child1.edge_weight('ftp', self.child2)
                btp = self.child2.edge_weight('btp', self.child1)
                return (ftp * btp) ** 0.5  # geometric mean

            else:
                form, degree = generalize
                if form == 'neighbor':
                    return neighbor_generalize(self, degree)
                else:
                    raise ValueError('Bad GENERALIZE parameter.')

    return Chunk


def neighbor_generalize(chunk, degree):
    similar_chunks = (chunk.model.get_chunk(predecessor, follower)
                      for predecessor in chunk.child2.predecessors
                      for follower in chunk.child1.followers)
    similar_chunks = [c for c in similar_chunks if c is not None]
    TEST['similar_chunks'] = similar_chunks

    # TODO make this 0 - 1
    gen_chunkiness = sum(chunk.chunkiness(generalize=False)
                         for chunk in similar_chunks)
    
    result =  (degree * gen_chunkiness + 
                (1-degree) * chunk.chunkiness(generalize=False))
    
    assert not np.isnan(result)
    return result
