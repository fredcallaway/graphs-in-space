"""

"""


from collections import Counter, OrderedDict
import time
from typing import Dict, List
from typed import typechecked
import numpy as np

import fuckit

import pcfg
import plotting
import utils
import vectors

SEMANTIC_TRANSFER = 0.2
LEARNING_RATE = 0.01
DECAY_RATE = 0.001

CHUNK_THRESHOLD = .5
MIN_NODE_COUNT = 3

FTP_PREFERENCE = .5  # TODO
MEMORY_SIZE = 4


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
        self.string = string
        self.idx = idx
        self.count = 0

        self.forward_edges = Counter()
        self.backward_edges = Counter()

        self.id_vector = id_vector
        self.semantic_vector = np.copy(self.id_vector)

    @property
    def context_vector(self) -> np.ndarray:
        """Represents this node in context."""
        return (self.semantic_vector * SEMANTIC_TRANSFER +
                self.id_vector * (1 - SEMANTIC_TRANSFER))

    @property
    def before_context_vector(self) -> np.ndarray:
        """Represents this node occurring before another node"""
        return np.roll(self.context_vector, 1)

    @property
    def after_context_vector(self) -> np.ndarray:
        """Represents this node occurring after another node"""
        return np.roll(self.context_vector, -1)  # TODO: parameterize this

    def decay(self) -> None:
        """Makes learned vector more similar to initial random vector."""
        self.semantic_vector += self.id_vector * DECAY_RATE

    @property
    def description(self) -> str:
        followers = [(n[0].string, n[1]) for n in self.forward_edges.most_common(5)]
        preceders = [(n[0].string, n[1]) for n in self.backward_edges.most_common(5)]
        return ('Node({self.string})\n'
                'Followed by: {followers}\n'
                'Preceded by: {preceders}').format(**locals())

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
    def __init__(self, numila, utterance) -> None:
        super(Parse, self).__init__()
        self.numila = numila

        for token in utterance:
            self.numila.decay()  # model's knowledge decays on every word it hears
            self.shift(token)
            self.update_weights(MEMORY_SIZE)
            if len(self) >= MEMORY_SIZE:  # fill memory before trying to chunk
                # possibly replace two odes in working memory with one node
                self.try_to_chunk(MEMORY_SIZE)

        # Process the tail end. We have to shrink memory_size to prevent
        # accessing elements that fell out of the 4 item memory window.
        for memory_size in range(MEMORY_SIZE-1, 1, -1):
            self.update_weights(memory_size)
            self.try_to_chunk(memory_size)


    def shift(self, token) -> None:
        LOG.debug('shift {token}'.format_map(locals()))
        node = self.numila[token]
        node.count += 1
        self.append(node)

    def update_weights(self, memory_size) -> None:
        memory = self[-memory_size:]

        # We have to make things a little more complicated to avoid
        # updating based on vectors changed in this round of updates.
        ftp_updates = {node: node.after_context_vector * LEARNING_RATE * FTP_PREFERENCE
                       for node in memory[1:]}
        btp_updates = {node: node.before_context_vector * LEARNING_RATE * (1 - FTP_PREFERENCE)
                       for node in memory[:-1]}

        for i in range(len(memory) - 1):
            node = memory[i]
            next_node = memory[i + 1]

            node.forward_edges[next_node] += 1
            next_node.backward_edges[node] += 1
            
            node.semantic_vector += ftp_updates[next_node]
            next_node.semantic_vector += btp_updates[node]

    def try_to_chunk(self, memory_size) -> None:
        """Attempts to combine two Nodes in memory into one Node.

        If no Node pairs form a chunk, then nothing happens. """
        memory = self[-memory_size:]
        if len(memory) < 2:
            return
        LOG.debug('memory_size = {memory_size}'.format_map(locals()))
        
        # invariant: idx is the index of the earlier Node under consideration
        chunkabilities = []
        for idx in range(len(memory) - 1):
            node = memory[idx]
            next_node = memory[idx+1]

            chunkability = self.numila.chunkability(node, next_node)
            LOG.debug('chunkability({node}, {next_node}) = {chunkability}'.format_map(locals()))
            chunkabilities.append(chunkability)
    
        best_idx = chunkabilities.index(max(chunkabilities))
        parse_idx = best_idx - len(memory)  # convert index in memory to index in parse
        assert parse_idx < -1  # must be an index from tail, and not the last element

        chunk = self.numila.get_chunk(self[parse_idx], self[parse_idx + 1])
        if chunk:
            # combine the two nodes into one chunk
            chunk.count += 1
            self[parse_idx] = chunk
            del self[parse_idx+1]

    def __str__(self):
        return super(Parse, self).__str__().replace(',', '')


class Numila(object):
    """The premier language acquisition model"""
    def __init__(self, size=100000) -> None:
        super(Numila, self,).__init__()
        # Each token gets an int ID which specifies its index
        # in self.nodes and self.activations.
        self.string_to_index = OrderedDict()  # type: Dict[str, int]
        self.nodes = []  # type: List[Node]
        self.activations = np.zeros(size)
        self.counts = np.zeros(size)

    def parse_utterance(self, utterance):
        if isinstance(utterance, str):
            utterance = utterance.split()
        return Parse(self, utterance)

    def create_token(self, token_string) -> Node:
        """Add a new base token to the graph."""
        id_vector = vectors.sparse()
        return self._create_node(token_string, id_vector)

    def create_chunk(self, node1, node2) -> Node:
        """Add a new chunk to the graph composed of node1 and node2."""
        chunk_string = '[{node1.string} {node2.string}]'.format_map(locals())
        id_vector = vectors.bind(node1.context_vector, node2.context_vector)
        return self._create_node(chunk_string, id_vector)

    def _create_node(self, node_string, id_vector) -> Node:
        idx = len(self.nodes)
        new_node = Node(self, node_string, idx, id_vector)
        self.nodes.append(new_node)
        self.string_to_index[node_string] = idx
        self.activations[idx] = 1.0
        return new_node

    def get_chunk(self, node1, node2) -> Node:
        """Returns a chunk of node1 and node2 if the chunk is in the graph.

        If the chunk doesn't exist, we check if it should be created. It is
        returned if it is created."""
        chunk_string = '[{node1.string} {node2.string}]'.format_map(locals())
        if chunk_string in self:
            return self[chunk_string]
        else:
            # consider making a new node
            if self.chunkability(node1, node2) > CHUNK_THRESHOLD:
                return self.create_chunk(node1, node2)
            else:
                return None

    def chunkability(self, node1, node2) -> float:
        """How well two nodes form a chunk."""

        # TODO check if the nodes already make a node
        # ftp looks at the filler identity vector, thus is parallel to decoding of BEAGLE
        ftp = vectors.cosine(node1.semantic_vector, node2.after_context_vector)
        btp = vectors.cosine(node1.before_context_vector, node2.semantic_vector)
        return (ftp + btp) / 2

    def predict(self, node1):
        """Returns the node most likely to follow node1"""
        chunkabilities = np.array([self.chunkability(node1, node2)
                                   for node2 in self.nodes])
        chunkabilities += 1.0  # because probabilites must be non-negative
        probs = chunkabilities / np.sum(chunkabilities)
        return np.random.choice(self.nodes, p=probs)

    def decay(self) -> None:
        for node in self.nodes:
            node.decay()

    def __getitem__(self, node_string) -> Node:
        if node_string in self.string_to_index:
            idx = self.string_to_index[node_string]
            return self.nodes[idx]
        else:
            if ' ' in node_string:
                raise ValueError('{node_string} is not in the graph.'.format_map(locals()))
            # node_string represents a new token, so we make a new Node for it
            return self.create_token(node_string)

    def __contains__(self, item) -> bool:
        return item in self.string_to_index

    # For introspection only: not used in the model itself.
    def similarity_matrix(self, round_to=None, num=None) -> "DataFrame":
        """A distance matrix of all nodes in the graph."""
        if not self.nodes:
            raise ValueError("Graph is empty, can't make distance matrix.")
        if num:
            #ind = np.argpartition(self.counts, -num)[-num:]
            raise NotImplementedError()
        sem_vecs = [node.semantic_vector for node in self.nodes]
        num_nodes = len(self.nodes)
        matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i, num_nodes):
                matrix[i,j] = matrix[j,i] = vectors.cosine(sem_vecs[i], sem_vecs[j])
        if round_to is not None:
            matrix = np.around(matrix, round_to)

        from pandas import DataFrame
        labels = self.string_to_index.keys()
        return DataFrame(matrix, 
                         columns=labels,
                         index=labels)


def word_sim(m, word1, word2):
    return vectors.cosine(m[word1].semantic_vector, m[word2].semantic_vector)


def cfg():
    m = Numila()

    for i, s in enumerate(pcfg.random_sentences('toy_pcfg2.txt', 100)):
        if i % 100 == 99:
            pass
            #plotting.distance_matrix(m.similarity_matrix())
        m.parse_utterance(s)


    for w in ['saw', 'the', 'telescope', '[my hill]']:
        with fuckit:
            print(w, '->',m.predict(m[w]))

    for s in pcfg.random_sentences('toy_pcfg2.txt', 5):
        parse = str(m.parse_utterance(s))
        print(parse)


def syl():
    m = Numila()
    for utt in utils.read_file('../PhillipsPearl_Corpora/English/test_sets/English_syllable_test1.txt', r'/| '):
        m.parse_utterance(utt)
    for utt in utils.read_file('../PhillipsPearl_Corpora/English/test_sets/English_syllable_test2.txt', r'/| ')[:100]:
        print(m.parse_utterance(utt))

    with fuckit:
        print(m.chunkability(m['vI'], m['di']))
        print(m.chunkability(m['di'], m['o']))
        print(m.chunkability(m['[vI di]'], m['o']))


def main() -> None:
    cfg()
    #syl()



if __name__ == '__main__':
    #print('\n')
    main()