import logging
import re
from typing import List
import time

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


def read_file(file_path, token_delim=' ', utt_delim='\n') -> List[List[str]]:
    utterances = []
    with open(file_path) as f:
        for utterance in re.split(utt_delim, f.read()):
            if token_delim:
                tokens = re.split(token_delim, utterance)
            else:
                tokens = list(utterance)  # split by character
            utterances.append(tokens)
    return utterances


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