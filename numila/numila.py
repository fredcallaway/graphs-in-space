from collections import Counter
import time


class Umila(object):
    """The premier language acquisition model"""
    def __init__(self, config_file):
        super(Umila, self).__init__()
        params = read_params(config_file)
        self.__dict__.update(params)

        self.memory = Memory()
        self.graph = {}

    def read(self, token):
        """Reads one token and performs all associated computation"""
        # print '-'
        self.memory
        self.parse_set.prune()
        self.parse_set.update_temporal_edges()
        self.graph.decay()

    def read_file(self, file_path, split=' '):
        with open(file_path) as f:
            for line in f:
                start = time.time()
                if split:
                    tokens = line.split(split)
                else:
                    tokens = list(line)  # split by charcater
                del tokens[-1]  # remove trailing \n
                for t in tokens:
                    self.read(t)
                print 'read %s tokens in %s seconds' % (len(tokens), time.time() - start)


class Memory(object):
    """The working memory of Umila"""
    def __init__(self):
        pass
        
        
        

class Node(object):
    """A Node in a graph"""
    def __init__(self):
        self.forward = None
        self.backward = None


class Token(Node):
    """An atomic input unit"""
    def __init__(self, string):
        super(Token, self).__init__()
        self._string = _string


class Slot(Node):
    """A distribution over Nodes"""
    def __init__(self):
        super(Slot, self).__init__()
        self.fillers = {}


class Chunk(Node):
    """A sequence of Tokens and Slots"""
    def __init__(self, elements):
        super(Chunk, self).__init__()
        self.elements = elements
