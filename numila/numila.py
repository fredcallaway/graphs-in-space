from collections import deque
import itertools
import numpy as np
from scipy import stats

import yaml

import utils

LOG = utils.get_logger(__name__, stream='WARNING', file='INFO')
TEST = {'on': False}


class Numila(object):
    """The premier language acquisition model."""
    def __init__(self, param_file='params.yml', **params):

        # Read default params from file, overwriting with keyword arguments.
        with open(param_file) as f:
            self.params = yaml.load(f.read())
        if not all(k in self.params for k in params):
            raise ValueError
        for k in params:
            if k not in self.params:
                raise ValueError(k + 'is not a valid parameter.')
        self.params.update(params)
        LOG.info('parameters:\n\n%s\n', 
                 yaml.dump(self.params, default_flow_style=False))

        if self.params['CHUNK_THRESHOLD'] is None:
            self.params['CHUNK_THRESHOLD'] = self.params['EXEMPLAR_THRESHOLD']

        # The GRAPH parameter determines which implementation of a Graph
        # this Numila instance should use. Thus, Numila is a class that
        # is parameterized by another class, similarly to how a functor
        # in OCaml is a module parameterized by another module.
        graph = self.params['GRAPH'].lower()
        if graph == 'holograph':
            from holograph import HoloGraph as Graph
        elif graph == 'probgraph':
            from probgraph import ProbGraph as Graph
        else:
            raise ValueError('Invalid GRAPH parameter: {}'.format(self.params['GRAPH']))
        
        self.graph = Graph(edges=['ftp', 'btp'], params=self.params)


    def parse_utterance(self, utterance, learn=True, verbose=False):
        """Returns a Parse of the given utterance."""
        self.graph.decay()
        if isinstance(utterance, str):
            utterance = utterance.split(' ')
        if self.params['ADD_BOUNDARIES']:
            utterance = ['#'] + utterance + ['#']
        return Parse(self, utterance, learn=learn, verbose=verbose)

    def fit(self, training_corpus, lap=None):
        with utils.Timer(print_func=LOG.warning) as timer:
            try:
                for count, utt in enumerate(training_corpus, 1):
                    self.parse_utterance(utt)
                    if lap and count % lap == 0:
                        timer.lap(count)
            except KeyboardInterrupt:
                pass  # allow interruption of training
            LOG.warning('Trained on %s utterances in %s seconds', 
                        count, timer.elapsed)
            return self

    def score(self, utt, type='chunk_ratio'):
        """Returns a grammaticality score for an utterance."""
        parse = self.parse_utterance(utt, learn=False)
        possible_chunks = len(parse.utterance) - 1
        
        chunk_ratio = parse.num_chunks / possible_chunks

        
        return chunk_ratio

    def create_node(self, string):
        node = self.graph.create_node(string)
        # Add extra links.
        node.followers = set()
        node.predecessors = set()
        return node

    def create_chunk(self, node1, node2):
        edges = {'btp': node1, 'ftp': node2} if self.params['BIND'] else {}
        node = self.graph.bind(node1, node2, edges=edges)
        # Add extra links.
        node.followers = set()
        node.predecessors = set()
        node.child1 = node1
        node.child2 = node2

        #print(node, self.graph.edge_counts['ftp'][node1.id_string][node2.id_string],
              #'/', sum(self.graph.edge_counts['ftp'][node1.id_string].values()))
        return node

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
                LOG.debug('Fixing a chunk node')
                node1 = self.graph[node1.id_string]
            if node2.id_string in self.graph and node2 is not self.graph[node2.id_string]:
                LOG.debug('Fixing a chunk node')
                node2 = self.graph[node2.id_string]
            chunk = self.create_chunk(node1, node2)
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

    def speak(self, words, verbose=False, return_chunk=False, preshuffled=False):
        """Returns the list of words ordered properly."""
        def get_node(token):
            try:
                return self.graph[token]
            except KeyError:
                LOG.debug('Unknown token while speaking: %s', token)
                return self.create_node(token)
        nodes = [get_node(w) for w in words]

        if not preshuffled:
            # In the case of a tie, the first pair is chosen, thus we shuffle
            # to make this effect random.
            np.random.shuffle(nodes)


        # Combine the two chunkiest nodes into a chunk until only one node left.
        while len(nodes) > 1:
            pairs = list(itertools.permutations(nodes, 2))
            best_pair = max(pairs, key=lambda pair: self.chunkiness(*pair))
            node1, node2 = best_pair
            chunk = self.get_chunk(node1, node2, stored_only=False)

            nodes.remove(node1)
            nodes.remove(node2)
            LOG.debug('\tchunk: %s', chunk)
            nodes.append(chunk)

        final = nodes[0]
        if return_chunk:
            return final
        else:
            return utils.flatten_parse(final)

    def chunkiness(self, node1, node2, generalize=None):
        """How well two nodes form a chunk.

        The geometric mean of forward transitional probability and
        bakward transitional probability.
        """

        if generalize is None:
            generalize = self.params['GENERALIZE']

        if not generalize:
            ftp = (self.graph.edge_weight('ftp', node1, node2) *
                   self.params['FTP_PREFERENCE'])
            btp = self.graph.edge_weight('btp', node2, node1)
            return (ftp * btp) ** 0.5  # geometric mean

        else:
            form, degree = generalize
            if form == 'neighbor':
                return self.neighbor_generalize(node1, node2, degree)
            elif form == 'full':
                return self.full_generalize(node1, node2, degree)
            else:
                raise ValueError('Bad GENERALIZE parameter.')

    def neighbor_generalize(self, node1, node2, degree):
        similar_chunks = (self.get_chunk(predecessor, follower)
                          for predecessor in node2.predecessors
                          for follower in node1.followers)
        similar_chunks = [c for c in similar_chunks if c is not None]
        TEST['similar_chunks'] = similar_chunks

        # TODO make this 0 - 1
        gen_chunkiness = sum(self.chunkiness(*chunk, generalize=False)
                             for chunk in similar_chunks)
        
        result =  (degree * gen_chunkiness + 
                   (1-degree) * self.chunkiness(node1, node2, generalize=False))
        
        assert not np.isnan(result)
        return result

    def full_generalize(self, node1, node2, degree):
        def make_gen_node(node):
            sims = [node.similarity(other_node)
                    for other_node in self.graph.nodes.values()]
            gen_node = self.graph.sum(self.graph.nodes.values(), weights=sims)
            return gen_node

        gnode1, gnode2 = map(make_gen_node, (node1, node2))
        gen_chunkiness = self.chunkiness(gnode1, gnode2, generalize=False)

        result = (degree * gen_chunkiness + 
                  (1-degree) * self.chunkiness(node1, node2, generalize=False))
        assert not np.isnan(result)
        return result



class Parse(list):
    """A parse of an utterance represented as a list of Nodes.

    The parse is computed upon intialization. This computation has side
    effects for the parent Numila instance (i.e. learning). The loop
    in __init__ is thus both the comprehension and learning algorithm.
    """
    def __init__(self, model, utterance, learn=True, verbose=False):
        super().__init__()
        self.model = model
        self.utterance = utterance
        self.graph = model.graph
        self.params = model.params
        self.learn = learn
        self.chunkinesses = []
        self.memory = deque(maxlen=self.params['MEMORY_SIZE'])

        LOG.debug('\nPARSING: %s', utterance)

        utterance = iter(utterance)

        # Fill up memory before trying to chunk.
        while len(self.memory) < self.params['MEMORY_SIZE']:
            token = next(utterance, None)
            if token is None:
                LOG.debug('break')
                break  # less than MEMORY_SIZE tokens in utterance
            LOG.debug('\nmemory = %s', self.memory)
            self.update_weights()
            self.shift(token)

        # Chunk, learn, and shift until we run out of tokens, at which point we
        # keep chunking until we reduce the utterance to one chunk or drop everything.
        while self.memory:
            LOG.debug('\nmemory = %s', self.memory)
            self.update_weights()
            success = self.try_to_chunk()
            if not success:
                # Couldn't make a chunk. We remove the oldest node 
                # to make room for a new one.
                self.append(self.memory.popleft())
            token = next(utterance, None)
            if token is not None:
                self.shift(token)

    @property
    def num_chunks(self):
        """The number of chunks made during this Parse."""
        return len(self.chunkinesses)

    @property
    def chunkedness(self):
        # TODO: total chunkedness
        between = [self.model.chunkiness(n1, n2)
                   for n1, n2 in utils.neighbors(self)]
        total = between + self.chunkinesses
        return stats.gmean(total)

    def shift(self, token):
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
            node = self.model.create_node(token)
            if self.learn:
                self.graph.add_node(node)

        self.memory.append(node)

    def update_weights(self):
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
        ftp_factor = self.params['LEARNING_RATE']
        btp_factor = self.params['LEARNING_RATE']

        # Increase the weight between every node in memory. Note that
        # some of these nodes may be chunks that are not in the graph. TODO
        for node1, node2 in utils.neighbors(self.memory):
            LOG.debug('  strengthen %s and %s', node1, node2)
            if node1.id_string in self.graph and node2.id_string in self.graph:
                self.graph.bump_edge('ftp', node1, node2, ftp_factor)
                self.graph.bump_edge('btp', node2, node1, btp_factor)

        if self.params['CHUNK_LEARNING']:
            # Incerase the weight between nodes that are in chunks in memory.
            ftp_factor *= self.params['CHUNK_LEARNING']
            btp_factor *= self.params['CHUNK_LEARNING']

            def update_chunk(chunk):
                """Recursively updates weights between elements of the chunk"""
                if not hasattr(chunk, 'chunkiness'):
                    return
                if chunk.child1.id_string in self.graph and chunk.child2.id_string in self.graph:
                    self.graph.bump_edge('ftp', chunk.child1, chunk.child2, ftp_factor)
                    self.graph.bump_edge('btp', chunk.child2, chunk.child1, btp_factor)
                update_chunk(chunk.child1)
                update_chunk(chunk.child2)

            for node in self.memory:
                update_chunk(node)

    def try_to_chunk(self):
        """Attempts to combine two Nodes in memory into one Node.

        Returns True for success, False for failure.
        """

        if len(self.memory) == 1:
            # We can't create a chunk when there's only one node left.
            # This can only happen while processing the tail, so we
            # must be done processing
            LOG.debug('done parsing')
            return False


        pairs = list(utils.neighbors(self.memory))
        chunkinesses = [self.model.chunkiness(node1, node2)
                        for node1, node2 in pairs]
        
        best_idx = np.argmax(chunkinesses)
        best_chunkiness = chunkinesses[best_idx]

        #chunks = [self.model.get_chunk(node1, node2, stored_only=False)
        #          for node1, node2 in utils.neighbors(self.memory)]
        #chunkinesses = [chunk.chunkiness() for chunk in chunks]

        #best_idx = np.argmax(chunkinesses)
        #best_chunk = chunks[best_idx]
        #best_chunkiness = chunkinesses[best_idx]

        if best_chunkiness >= self.params['CHUNK_THRESHOLD']:  # TODO chunk threshold
            best_chunk = self.model.get_chunk(*pairs[best_idx], stored_only=False)
            # Replace the two nodes in memory with one chunk.
            self.memory[best_idx] = best_chunk
            del self.memory[best_idx+1]
            self.chunkinesses.append(best_chunkiness)

            # Add the chunk to the graph if it exceeds a threshold chunkiness.
            if (self.learn and
                best_chunk.id_string not in self.graph and
                best_chunkiness > self.params['EXEMPLAR_THRESHOLD']):
                    self.model.add_chunk(best_chunk)      
            LOG.debug('create chunk: %s', best_chunk)
            return True
        else:  # can't make a chunk
            LOG.debug('no chunk created')
            return False

    def __repr__(self):
        return super().__repr__().replace(',', ' |')[1:-1]

    def __str__(self):
        return self.__repr__()




