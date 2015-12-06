import logging
import re
from typing import List
import time
import itertools

def neighbors(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def get_logger(name, stream='WARNING', file='INFO'):
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
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


def read_corpus(file, token_delim=' ', utt_delim='\n') -> List[List[str]]:
    with open(file) as f:
        for utterance in re.split(utt_delim, f.read()):
            if token_delim:
                tokens = re.split(token_delim, utterance)
            else:
                tokens = list(utterance)  # split by character
            yield tokens


class Timer(object):
    """Context manager timer"""
    def __init__(self,name):
        self.name = name
    def __enter__(self):
        self.start = time.time()
    def __exit__(self,ty,val,tb):
        end = time.time()
        print("%s : %0.3f seconds" % (self.name, end-self.start))
        return False


def flatten(lst):
    """Returns a flat version of a list of (lists of (lists of...)) lists"""
    assert lst is not None
    def iterable(obj): 
        return hasattr(obj, '__iter__') and not isinstance(obj, str)

    flat_lists = [flatten(elem) if iterable(elem) else [elem]
                  for elem in lst]
    return list(itertools.chain.from_iterable(flat_lists))