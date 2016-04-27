import itertools
import numpy as np

import yaml
import utils

fmt = utils.literal


class Numila(object):
    """The premier language acquisition model."""
    def __init__(self, param_file='params.yml', name='numila', log_stream='WARNING',
                 log_file='WARNING', **params):
        self.name = name
        self.log = utils.get_logger(name, stream=log_stream, file=log_file)

        # Read default params from file, overwriting with keyword arguments.
        with open(param_file) as f:
            self.params = yaml.load(f.read())
        if not all(k in self.params for k in params):
            raise ValueError
        for k in params:
            if k not in self.params:
                raise ValueError(k + 'is not a valid parameter.')
        self.params.update(params)
        self.log.info('parameters:\n\n%s\n', 
                 yaml.dump(self.params, default_flow_style=False))

        # The GRAPH parameter determines which implementation of a Graph
        # this Numila instance should use. Thus, Numila is a class that
        # is parameterized by another class, similarly to how a functor
        # in OCaml is a module parameterized by another module.
        graph = self.params['GRAPH'].lower()
        if graph.startswith('holo'):
            from holograph import HoloGraph as Graph
        elif graph.startswith('prob'):
            from probgraph import ProbGraph as Graph
        else:
            raise ValueError('Invalid GRAPH parameter: {}'.format(self.params['GRAPH']))
        self.graph = Graph(edges=['ftp', 'btp'], **self.params)
        
        # Same deal for Parse.
        parse = self.params['PARSE'].lower()
        if parse == 'greedy':
            from greedy_parse import GreedyParse as Parse
        elif parse == 'full':
            from full_parse import FullParse as Parse
            if self.graph.HIERARCHICAL:
                self.log.warning('FullParse can only be used with non-hierarchical merge')
                self.graph.HIERARCHICAL = False
        else:
            raise ValueError('Invalid PARSE parameter: {}'.format(self.params['PARSE']))
        self.Parse = Parse

        self._debug = {'speak_chunks': 0}

    @property
    def chunk_threshold(self):
        pass # one day...        


    def parse(self, utterance, learn=True):
        """Parses the utterance and returns the result."""
        if self.params['DECAY']:
            self.graph.decay()
        if isinstance(utterance, str):
            utterance = utterance.split(' ')
        if self.params['ADD_BOUNDARIES']:
            utterance = ['ø'] + utterance + ['ø']
        return self.Parse(self, utterance, learn=learn)

    def fit(self, training_corpus, lap=None):
        """Trains the model on a training corpus."""
        with utils.Timer(print_func=None) as timer:
            try:
                for count, utt in enumerate(training_corpus, 1):
                    self.parse(utt)
                    if lap and count % lap == 0:
                        timer.lap(count)
            except KeyboardInterrupt:
                pass  # allow interruption of training
            self.log.warning('Trained on %s utterances in %s seconds', 
                        count, timer.elapsed)
            return self

    def score(self, utt, **kwargs):
        """Returns a grammaticality score for an utterance."""
        return self.parse(utt, learn=False).score(**kwargs)

    def map_score(self, utts, **kwargs):
        with utils.Timer(print_func=None) as timer:
            result = [self.score(u, **kwargs) for u in utts]
        self.log.warning('Scored %s utterances in %s seconds', 
                         len(utts), timer.elapsed)
        return result

    @utils.contract(lambda x: 0 <= x <= 1)
    def chunkiness(self, node1, node2):
        """How well two nodes form a chunk.

        The geometric mean of forward transitional probability and
        backward transitional probability.
        """
        ftp_weight = 1
        btp_weight = self.params['BTP_PREFERENCE']
        if btp_weight == 'only':
            ftp_weight, btp_weight = 0, 1
        generalize = self.params['GENERALIZE']

        ftp = node1.edge_weight(node2, 'ftp', generalize=generalize)
        btp = node2.edge_weight(node1, 'btp', generalize=generalize)
        sum_weights = btp_weight + ftp_weight
        gmean = (ftp ** ftp_weight * btp ** btp_weight) ** (1 / sum_weights)

        return gmean

    def get_chunk(self, node1, node2, *, create=False, add=False):
        """Returns a chunk of node1 and node2 if the chunk is in the graph.

        If `create` is False, we only return the desired chunk if it
        has been stored as an exemplar in the graph. Otherwise, we
        always return a chunk, creating it if necessary.

        If the chunk doesn't exist, we check if the pair is chunkable
        enough for a new chunk to be created. If so, the new chunk is returned.
        """
        existing_chunk = self.graph.get_chunk(node1, node2)
        if existing_chunk:
            return existing_chunk

        assert not (node1.id_string == 'ø' or node2.id_string == 'ø')
            
        if create:
            if node1.id_string in self.graph and node1 is not self.graph[node1.id_string]:
                self.log.debug('Fixing a chunk node')
                node1 = self.graph[node1.id_string]
            if node2.id_string in self.graph and node2 is not self.graph[node2.id_string]:
                self.log.debug('Fixing a chunk node')
                node2 = self.graph[node2.id_string]
            
            chunk = self.graph.bind(node1, node2)
            if add:
                self.graph.add(chunk)
            return chunk

    def add_chunk(self, chunk):
        self.graph.add(chunk)
        self.log.debug('new chunk: %s', chunk)

    def speak(self, words, verbose=False, return_flat=True, 
              preshuffled=False, order_func=None):
        """Returns the list of words ordered properly."""

        # Get all the base token nodes.
        def get_node(token):
            try:
                return self.graph[token]
            except KeyError:
                self.log.debug('Unknown token while speaking: %s', token)
                return self.graph.create_node(token)
        nodes = [get_node(w) for w in words]

        if not preshuffled:
            # In the case of a tie, the first pair is chosen, thus we shuffle
            # to make this effect random.
            np.random.shuffle(nodes)

        # Convert as many nodes as possible into chunks by combining
        # the two chunkiest nodes into a chunk until can't chunk again.
        while len(nodes) > 1:
            self.log.debug('nodes: %s', nodes)
            pairs = list(itertools.permutations(nodes, 2))
            best_pair = max(pairs, key=lambda pair: self.chunkiness(*pair))
            node1, node2 = best_pair
            chunk = self.get_chunk(node1, node2, create=False)
            self.log.debug('chunk: %s', chunk)
            if not chunk:
                break

            nodes.remove(node1)
            nodes.remove(node2)
            nodes.append(chunk)

        self._debug['speak_chunks'] = sum(1 for n in nodes if n.children)

        if order_func is None:  # ordering function is a parameter
            order_func = {'markov': self._order_markov,
                          'outward': self._order_outward}[self.params['SPEAK']]
        utterance = list(order_func(nodes))

        if return_flat:
            return utils.flatten_parse(utterance)
        else:
            return utterance

    def _order_markov(self, nodes):
        last_node = self.graph['ø']
        while nodes:
            #next_node = max(nodes, key=lambda n: self.chunkiness(last_node, n))
            best_idx = np.argmax([self.chunkiness(last_node, n) for n in nodes])
            next_node = nodes.pop(best_idx)
            yield next_node
            last_node = next_node

    def _order_outward(self, nodes):
        # most_common = max(nodes, key=node.weight)  # TODO
        utterance = [nodes.pop(0)]  
        while nodes:
            # Add a node to the beginning or end of the utterance.
            begin_chunkinesses = [self.chunkiness(n, utterance[0])
                                  for n in nodes]
            end_chunkinesses = [self.chunkiness(utterance[-1], n)
                                for n in nodes]
            
            best_idx = np.argmax(begin_chunkinesses + end_chunkinesses)
            if best_idx >= len(nodes):
                utterance.append(nodes.pop(best_idx % len(nodes)))
            else:
                utterance.insert(0, nodes.pop(best_idx))
        return utterance


if __name__ == '__main__':
    model = Numila()
    corpus = ['a b c'] * 3
    model.fit(corpus)
    model.speak(list('abbcc'))
