import itertools
import numpy as np
from scipy import stats

import yaml

import utils
fmt = utils.literal
from parse import Parse

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


    def parse(self, utterance, learn=True, verbose=False):
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
                    self.parse(utt)
                    if lap and count % lap == 0:
                        timer.lap(count)
            except KeyboardInterrupt:
                pass  # allow interruption of training
            LOG.warning('Trained on %s utterances in %s seconds', 
                        count, timer.elapsed)
            return self

    def score(self, utt, ratio=0, freebie=-1):
        """Returns a grammaticality score for an utterance."""
        parse = self.parse(utt, learn=False)


        def chunk_ratio():
            possible_chunks = len(parse.utterance) - 1
            return parse.num_chunks / possible_chunks

        def chunkiness():
            between = [self.chunkiness(n1, n2)
                       for n1, n2 in utils.neighbors(parse)]
            within = [max(chunkiness, freebie) for chunkiness in parse.chunkinesses]    
            total = np.array(between + within)
            total += .001  # smoothing
            return stats.gmean(total)

        return chunk_ratio() * ratio + chunkiness() * (1-ratio)

    def map_score(self, utts, **kwargs):
        return [self.score(u, **kwargs) for u in utts]

    def create_node(self, string):
        node = self.graph.create_node(string)
        # Add extra links.
        node.followers = set()
        node.predecessors = set()
        return node

    def create_chunk(self, node1, node2):
        edges = self.params['BIND'] and {'btp': node1, 'ftp': node2}
        node = self.graph.bind(node1, node2, edges=edges)
        
        # Add extra links for neighbor generalize algorithm.
        node.followers = set()
        node.predecessors = set()

        return node

    def get_chunk(self, node1, node2, stored_only=True):
        """Returns a chunk of node1 and node2 if the chunk is in the graph.

        If stored_only is True, we only return the desired chunk if it
        has been stored as an exemplar in the graph. Otherwise, we
        always return a chunk, creating it if necessary.

        If the chunk doesn't exist, we check if the pair is chunkable
        enough for a new chunk to be created. If so, the new chunk is returned.
        """
        chunk_id_string = fmt('[{node1.id_string} {node2.id_string}]')
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

    @utils.contract(lambda x: 0 <= x <= 1)
    def chunkiness(self, node1, node2, generalize=None):
        """How well two nodes form a chunk.

        The geometric mean of forward transitional probability and
        bakward transitional probability.
        """

        if generalize is None:
            generalize = self.params['GENERALIZE']

        if not generalize:
            ftp_weight = self.params['FTP_PREFERENCE']
            btp_weight = 1
            ftp = node1.edge_weight(node2, 'ftp') ** ftp_weight
            btp = node2.edge_weight(node1, 'btp')
            sum_weights = btp_weight + ftp_weight
            result = (ftp * btp) ** (1 / sum_weights)  # geometric mean
            assert not np.isnan(result)
            return result

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
                    for other_node in self.graph.nodes]
            gen_node = self.graph.sum(self.graph.nodes, weights=sims)
            return gen_node

        gnode1, gnode2 = map(make_gen_node, (node1, node2))
        gen_chunkiness = self.chunkiness(gnode1, gnode2, generalize=False)

        result = (degree * gen_chunkiness + 
                  (1-degree) * self.chunkiness(node1, node2, generalize=False))
        assert not np.isnan(result)
        return result

