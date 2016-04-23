import numpy as np
from functools import lru_cache
from scipy import stats

import utils


def make_list(generator_func):
    from functools import wraps
    @wraps(generator_func)
    def wrapper(*args, **kwargs):
        return list(generator_func(*args, **kwargs))
    return wrapper



class FullParse(object):
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

        self.log = model.log
        self.log.debug('')
        self.log.debug('PARSING: %s', utterance)

        # Bump every pair of adjacent nodes in the utterance.
        for divide in range(1, len(utterance)):
            self.log.debug('divide: %s', divide)
            nodes_ending = list(self.get_nodes(divide, end=True))
            nodes_starting = list(self.get_nodes(divide, end=False))
            for n1 in nodes_ending:
                for n2 in nodes_starting:
                    self.bump(n1, n2)
                    self.try_to_chunk(n1, n2)

    def score(self):
        """The sum of the score of every path through the utterance.

        A score of one path is the geometric mean of the transitional
        probabilities in that path."""

        # We use a recursive algorithm with cacheing. See scored_paths
        all_scores = (np.exp(scored_path[0]) ** (1/len(self.utterance))
                      for scored_path in self.scored_paths(0, len(self.utterance)))
        return sum(all_scores)

    @lru_cache(None)
    @make_list
    def scored_paths(self, start, end):
        """Yields (score, path) tuples that span from start to end."""
        starting_here = self.get_nodes(start)
        for node in starting_here:
            next_pos = start + len(node)
            if next_pos == end:
                # Base case: we're at the end.
                yield (0, [node])
                return

            for score, path in self.scored_paths(next_pos, end):
                transition = self.model.chunkiness(node, path[0]) + .001  # smoothing
                full_score = score + np.log(transition)
                full_path = [node] + path
                yield (full_score, full_path)
                    

    @lru_cache(None)
    @make_list
    def get_nodes(self, pos, end=False):
        #print('get_nodes ', pos, end)
        if end:
            candidates = [self.utterance[start:pos] for start in range(pos)]
        else:
            candidates = [self.utterance[pos:end] 
                          for end in range(pos+1, len(self.utterance)+1)]
        for token_lst in candidates:
            #self.log.debug('consider %s', ' '.join(token_lst))
            if len(token_lst) == 1:
                node_string = token_lst[0]
                if node_string not in self.graph:
                    self.graph.add(node_string)
            else:
                node_string = ' '.join(('[', *token_lst, ']'))
            if node_string in self.graph:
                #self.log.debug('yield %s', node_string)
                yield self.graph[node_string]

    def bump(self, n1, n2):
        if not self.learn:
            return
        self.log.debug('  bump %s -> %s', n1, n2)
        n1.bump_edge(n2, 'ftp', self.params['LEARNING_RATE'])
        n2.bump_edge(n1, 'btp', self.params['LEARNING_RATE'])

    def try_to_chunk(self, n1, n2):
        if not self.learn:
            return
        if n1.id_string == '#' or n2.id_string == '#':
            return
        if self.graph._id_string((n1, n2)) not in self.graph:
            if self.model.chunkiness(n1, n2) > self.params['CHUNK_THRESHOLD']:
                chunk = self.graph.bind(n1, n2)
                self.graph.add(chunk)



def test_parse():
    import numila
    model = numila.Numila(PARSE='full', ADD_BOUNDARIES=False)
    #model.parse('the dog ate')
    #model.parse('the dog ate a steak')
    model.parse('I know the dog ate a steak')
    #model.parse('I know the dog ate a steak')
    #model.parse('I know the dog ate a steak')
    #print(*model.parse('I know the dog ate a steak').routes(0, 7), sep='\n')
    print(model.parse('I know the dog ate a steak', learn=False).score())
    print(model.parse('I know the dog ate', learn=False).score())
    print(model.parse('the dog ate a steak', learn=False).score())
    print(model.parse('the dog ate a', learn=False).score())
    print(model.parse('the dog ate', learn=False).score())
    print('-' * 50)
    print(model.parse('the ate dog', learn=False).score())
    print(model.parse('the dog ate ', learn=False).score())
    print(model.parse('the the', learn=False).score())
    print(model.parse('the the the the', learn=False).score())



if __name__ == '__main__':
    test_parse()