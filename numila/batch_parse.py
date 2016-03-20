import utils
from collections import deque
import numpy as np
from functools import lru_cache
from scipy import stats



def make_list(generator_func):
    from functools import wraps
    @wraps(generator_func)
    def wrapper(*args, **kwargs):
        return list(generator_func(*args, **kwargs))
    return wrapper



class Parse(object):
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
        for divide in range(1,len(utterance)):
            self.log.debug('divide: %s', divide)
            nodes_ending = list(self.get_nodes(divide, end=True))
            nodes_starting = list(self.get_nodes(divide, end=False))
            for n1 in nodes_ending:
                for n2 in nodes_starting:
                    self.bump(n1, n2)

    def score(self):
        """The sum of the score of every path through the utterance.

        A score of one path is the geometric mean of the transitional
        probabilities in that path."""
        return sum(np.product([self.model.chunkiness(*pair)
                                for pair in utils.neighbors(route)])
                    for route in self.routes(0, len(self.utterance)))

        
    @lru_cache(None)
    @make_list
    def routes(self, start, end):
        """Yields lists of nodes that span from start to end."""
        #print('routes', start, end)
        if start == end:
            yield []
            return
        nodes = self.get_nodes(start)
        for node in nodes:
            next_pos = start + len(node)
            for route in self.routes(next_pos, end):
                yield [node] + route


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
        if self.graph._id_string((n1, n2)) not in self.graph:
            if self.model.chunkiness(n1, n2) > self.params['CHUNK_THRESHOLD']:
                chunk = self.graph.bind(n1, n2)
                self.graph.add(chunk)



def test_parse():
    import numila
    model = numila.Numila(PARSE='batch')
    #model.parse('the dog ate')
    #model.parse('the dog ate a steak')
    model.parse('I know the dog ate a steak')
    model.parse('I know the dog ate a steak')
    model.parse('I know the dog ate a steak')
    #print(*model.parse('I know the dog ate a steak').routes(0, 7), sep='\n')
    print(model.parse('I know the dog ate a steak', learn=False).score())
    print(model.parse('I know the dog ate', learn=False).score())
    print(model.parse('the dog ate a steak', learn=False).score())
    print(model.parse('the dog ate a', learn=False).score())
    print(model.parse('the dog ate', learn=False).score())
    #print(model.parse('the dog', learn=False).score())


if __name__ == '__main__':
    test_parse()