import logging
import re
from typing import List
import time
import itertools



def make(return_type):
    """Decorator that makes a generator function return another type.

    Generalizes the @vectorized decorator, as described here:
    http://www.scipy-lectures.org/advanced/advanced_python/#a-while-loop-removing-decorator

    Example:
        >>> import pandas as pd
        >>> @make(pd.DataFrame)
        >>> def exponent_df(n):
                for i in range(n):
                    yield {'a': i, 'b': i ** 2, 'c': i ** 3}

        >>> exponent_df(4)
           a  b   c
        0  0  0   0
        1  1  1   1
        2  2  4   8
        3  3  9  27

    """
    from functools import update_wrapper
    def maker(generator_func):
        def wrapper(*args, **kwargs):
            return return_type(generator_func(*args, **kwargs))
        return update_wrapper(wrapper, generator_func)
    return maker

def neighbors(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def get_logger(name, stream='WARNING', file='INFO'):
    log = logging.getLogger(name)
    
    format_ = '[%(name)s : %(levelname)s]\t%(message)s'
    if stream and not any(isinstance(h, logging.StreamHandler) for h in log.handlers):
        printer = logging.StreamHandler()
        printer.setLevel(getattr(logging, stream))
        printer.setFormatter(logging.Formatter(fmt=format_))
        log.addHandler(printer)

    if file and not any(isinstance(h, logging.FileHandler) for h in log.handlers):
        filer = logging.FileHandler('log.txt')
        filer.setLevel(getattr(logging, file))
        filer.setFormatter(logging.Formatter(fmt=format_))
        log.addHandler(filer)

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
    """Context manager timer"""
    def __init__(self, name, print_func=print):
        self.name = name
        self.print_func = print_func
    
    def __enter__(self):
        self.start = time.time()
    
    def __exit__(self,ty,val,tb):
        end = time.time()
        self.print_func("%s: %0.3f seconds" % 
                        (self.name, end-self.start))
        return False


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

