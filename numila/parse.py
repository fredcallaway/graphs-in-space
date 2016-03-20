import utils
from collections import deque
import numpy as np
from scipy import stats

class Parse(list):
    """A parse of an utterance represented as a list of Nodes.

    The parse is computed upon intialization. This computation has side
    effects for the parent Numila instance (i.e. learning). The loop
    in __init__ is thus both the comprehension and learning algorithm.
    """
    def __init__(self, model, utterance, learn=True) -> None:
        super().__init__()
        self.model = model
        self.utterance = utterance
        self.graph = model.graph
        self.params = model.params
        self.learn = learn
        self.chunkinesses = []
        self.memory = deque(maxlen=self.params['MEMORY_SIZE'])

        self.log = model.log
        self.log.debug('')
        self.log.debug('PARSING: %s', utterance)

        utterance = iter(utterance)

        # Fill up memory before trying to chunk.
        while len(self.memory) < self.params['MEMORY_SIZE']:
            token = next(utterance, None)
            if token is None:
                self.log.debug('Break early.')
                break  # less than MEMORY_SIZE tokens in utterance
            self.shift(token)
            self.log.debug('memory = %s', self.memory)
            self.update_weights(-1)


        # Chunk, learn, and shift until we run out of tokens, at which point we
        # keep chunking until we reduce the utterance to one chunk or drop everything.
        self.log.debug('Begin chunking.')
        while self.memory:
            chunk_idx = self.try_to_chunk()
            if chunk_idx is None:
                # Couldn't make a chunk. We remove the oldest node 
                # to make room for a new one.
                oldest = self.memory.popleft()
                self.append(oldest)
                self.log.debug('dropped %s', oldest)
            else:
                self.update_weights(chunk_idx, 'new')
            token = next(utterance, None)
            if token is not None:
                self.shift(token)
                self.update_weights(-1, 'new')
            self.log.debug('memory = %s', self.memory)
            self.update_weights(-1, 'old')

    def score(self, ratio=0, freebie=-1):

        def chunk_ratio():
            possible_chunks = len(self.utterance) - 1
            return len(self.chunkinesses) / possible_chunks

        def gmean_chunkiness():
            between = [self.model.chunkiness(n1, n2)
                       for n1, n2 in utils.neighbors(self)]
            within = [max(chunkiness, freebie) for chunkiness in self.chunkinesses]    
            total = np.array(between + within)
            total += .001  # smoothing
            return stats.gmean(total)

        return chunk_ratio() * ratio + gmean_chunkiness() * (1-ratio)

    def shift(self, token) -> None:
        """Adds a token to memory.

        This method can only be called when there is an empty slot in
        memory. If token has never been seen, this method creates a new
        node in the graph for it.
        """
        assert len(self.memory) < self.memory.maxlen
        self.log.debug('shift: %s', token)

        try:
            node = self.graph[token]
        except KeyError:  # a new token
            node = self.model.create_node(token)
            if self.learn:
                self.graph.add(node)

        self.memory.append(node)

    def update_weights(self, new_idx=None, learn_mode=None) -> None:
        """Strengthens the connection between every adjacent pair of nodes in memory.

        Additionally, we increase the connection between nodes that are in a 
        chunk in memory. For a pair ([the big], dog) we increase the weight of 
        the forward edge, [the big] -> dog, and the backward edge, dog -> [the big],
        as well as the forward edge, the -> big, and backward edge, big -> the.
        """
        if not self.learn:
            return
        if learn_mode and learn_mode != self.params['LEARN_MODE']:
            # This update should't occur given the current LEARN_MODE parameter.
            return

        # These factors determine how much we should increase the weight
        # of each type of edge.
        # FIXME: only use FTP_PREFERENCE in one place.
        ftp_factor = self.params['LEARNING_RATE']  #  * self.params['FTP_PREFERENCE'] 
        btp_factor = self.params['LEARNING_RATE']

        if self.params['LEARN_MODE'] == 'new':
            to_bump = self.adjacent(new_idx)
        else:
            to_bump = utils.neighbors(self.memory)

        for node1, node2 in to_bump:
            self.log.debug('  bump %s -> %s', node1, node2)
            if node1.id_string in self.graph and node2.id_string in self.graph:
                node1.bump_edge(node2, 'ftp', ftp_factor)
                node2.bump_edge(node1, 'btp', btp_factor)

        if self.params['CHUNK_LEARNING']:
            # Incerase the weight between nodes that are in chunks in memory.
            ftp_factor *= self.params['CHUNK_LEARNING']
            btp_factor *= self.params['CHUNK_LEARNING']

            def update_chunk(chunk):
                """Recursively updates weights between elements of the chunk"""
                if not hasattr(chunk, 'chunkiness'):
                    return
                # if chunk.child1.id_string in self.graph 
                # and chunk.child2.id_string in self.graph:
                chunk.child1.bump_edge(chunk.child2, 'ftp', ftp_factor)
                chunk.child2.bump_edge(chunk.child1, 'btp', btp_factor)
                update_chunk(chunk.child1)
                update_chunk(chunk.child2)

            for node in self.memory:
                update_chunk(node)

    def try_to_chunk(self) -> None:
        """Attempts to combine two Nodes in memory into one Node.

        Returns True for success, False for failure.
        """

        if len(self.memory) == 1:
            # We can't create a chunk when there's only one node left.
            # This can only happen while processing the tail, so we
            # must be done processing
            self.log.debug('done parsing')
            return None


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

        if best_chunkiness > self.params['CHUNK_THRESHOLD']:  # TODO chunk threshold
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
            self.log.debug('create chunk: %s', best_chunk)
            return best_idx
        else:  # can't make a chunk
            self.log.debug('no chunk created')
            return None

    def adjacent(self, idx):
        """Returns nodes adjacent to the given index in memory."""
        depth = self.params['DEPTH']
        idx = idx % len(self.memory)
        node = self.memory[idx]
        left = idx - 1
        if left >= 0:
            other_node = self.memory[left]
            # Yield all children on the right side.
            for _ in range(depth + 1):
                yield (other_node, node)
                other_node = other_node.child2
                if not other_node: break

        right = idx + 1
        if right < len(self.memory):
            other_node = self.memory[right]
            for _ in range(depth + 1):
                yield (node, other_node)
                other_node = other_node.child1
                if not other_node: break

    def __repr__(self):
        return super().__repr__().replace(',', ' |')[1:-1]

    def __str__(self):
        return self.__repr__()



def test_parse():
    import numila
    model = numila.Numila()
    #self.log.handlers[0].setLevel('DEBUG')
    #self.log.setLevel('DEBUG')
    model.parse('the dog ate')
    model.parse('the dog ate a steak')
    model.parse('I know the dog ate a steak')

if __name__ == '__main__':
    test_parse()