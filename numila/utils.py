import logging
import re
from typing import List
import time
import itertools
from contextlib import contextmanager
from functools import wraps

@contextmanager
def capture_logging(logger, level='DEBUG'):
    """Captures log messages at or above the given level on the given logger.

    Context object is a function that returns the captured log as a string.

    Example:
        >>> with capture_logging('mylogger', 'INFO') as logged:
                function_that_logs_to_mylogger()
        >>> print(logged())
    """
    import io
    import logging
    if isinstance(logger, str):
        logger = logging.getLogger(logger)

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)

    old_level = logger.level
    logger.setLevel(level)
    logger.addHandler(handler)

    class LogStream:
        def __call__(self):
            try:
                handler.flush()
                val = stream.getvalue()
                self.val = val
                return val
            except ValueError:
                return self.val
    
    result = LogStream()
    yield result
    result()
    stream.close()
    logger.removeHandler(handler)
    logger.setLevel(old_level)


def neighbors(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def get_logger(name, stream='WARNING', file='INFO'):
    log = logging.getLogger(name)
    
    format_ = '[%(name)s]\t%(message)s'
    if stream and not any(isinstance(h, logging.StreamHandler) for h in log.handlers):
        printer = logging.StreamHandler()
        printer.setLevel(stream)
        printer.setFormatter(logging.Formatter(fmt=format_))
        log.addHandler(printer)

    if file and not any(isinstance(h, logging.FileHandler) for h in log.handlers):
        filer = logging.FileHandler('log.txt')
        filer.setLevel(file)
        filer.setFormatter(logging.Formatter(fmt=format_))
        log.addHandler(filer)

    log.setLevel(min(filer.level, printer.level))
    return log


def take_unique(seq, n):
    """Returns set of next n unique items in a sequence."""
    result = set()
    while len(result) < n:
        x = next(seq)
        if isinstance(x, list):
            x = tuple(x)
        if x not in result:
            result.add(x)
    return result


def read_corpus(file, token_delim=' ', utt_delim='\n', num_utterances=None) -> List[List[str]]:
    """A list of lists of tokens in a corpus"""
    with open(file) as f:
        for idx, utterance in enumerate(re.split(utt_delim, f.read())):
            if idx == num_utterances:
                break
            if token_delim:
                tokens = re.split(token_delim, utterance)
            else:
                tokens = list(utterance)  # split by character
            yield tokens

def cfg_corpus(n=None):
    corpus = read_corpus('corpora/toy2.txt', num_utterances=n)
    return corpus

def syl_corpus(n=None):
    corpus = read_corpus('../PhillipsPearl_Corpora/English/English-syl.txt',
                               token_delim=r'/| ', num_utterances=n)
    return corpus


class Timer(object):
    """A context manager which times the block it surrounds.

    Args:
        name (str): The name of the timer for time messages
        print_func (callable): The function used to print messages
          e.g. logging.debug

    Based heavily on https://github.com/brouberol/contexttimer

    Example:
    >>> with Timer('Busy') as timer:
            [i**2 for i in range(100000)]
            timer.lap()
            [i**2 for i in range(100000)]
            timer.lap()
            [i**2 for i in range(100000)]
    Busy (0): 0.069 seconds
    Busy (1): 0.126 seconds
    Busy (total): 0.176 seconds
    """
    def __init__(self, name='Timer', print_func=print):
        self.name = name
        self.print_func = print_func or (lambda *args: None)
        self._lap_idx = 0
        self._done = None

    @property
    def elapsed(self):
        current = self._done or time.time()
        return current - self.start

    def lap(self, label=None):
        if label is None:
            label = self._lap_idx
        self._lap_idx += 1
        self.print_func("%s (%s): %0.3f seconds" % (self.name, label, self.elapsed))

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self,ty,val,tb):
        self.lap('total')


def flatten(lst):
    """Returns a flat version of a list of (lists of (lists of...)) lists"""
    assert lst is not None
    def iterable(obj): 
        return hasattr(obj, '__iter__') and not isinstance(obj, str)

    flat_lists = [flatten(elem) if iterable(elem) else [elem]
                  for elem in lst]
    return list(itertools.chain.from_iterable(flat_lists))


def flatten_parse(parse):
    """A flat list of the words in a parse."""
    no_brackets = re.sub(r'[()[\]]', '', str(parse))
    return no_brackets.split(' ')


def generate_args(params):
    """Returns all possible permutations of params in parameters

    parmeters must be a list of (str, list) tuples, where str is
    the key and list is a list of values. A list of dictionaries
    is returned.
    """
    keys, valss = zip(*params)
    product = itertools.product(*valss)
    return [{k: v for k, v in zip(keys, vals)} for vals in product]

