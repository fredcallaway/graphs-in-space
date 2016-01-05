from numila import Numila
import numpy as np
import main
import utils

#def test_speaking(numila):
#    words = 'the hill ate my cookie'.split()
#    utterance = str(numila.speak(words))
#    assert all(w in utterance for w in words)
#    no_brackets = utterance.replace('[', '').replace(']', '')
#    assert len(no_brackets.split()) == len(words)
#    print(utterance)


def test_simple():
    corpus = utils.read_corpus('corpora/test.txt')
    numila = Numila(DIM=1000, ADD_BOUNDARIES=False).fit(corpus)
    parse = numila.parse_utterance('the boy ate the boy')
    assert parse


def test_generalization():
    return
    numila = Numila(DIM=100, GENERALIZE=0.1, ADD_BOUNDARIES=False).fit(main.cfg_corpus(n=100))
    #basic_tester(numila)


def basic_tester(numila):
    parse = numila.parse_utterance('the boy ate')
    
    equal = parse.log_chunkiness == log_chunkiness(parse)
    # We check explicitly for np.nan because np.nan != np.nan
    both_nan = np.isnan(parse.log_chunkiness) or np.isnan(log_chunkiness(parse))
    assert equal or both_nan

    assert not np.isnan(parse.log_chunkiness)

    print('\nFIRST parse of foo bar')
    numila.parse_utterance('foo bar', verbose=True)
    #numila.parse_utterance('foo bar')
    #numila.parse_utterance('foo bar')
    #numila.parse_utterance('foo bar')
    print('\nSECOND parse of foo bar')
    foobar = numila.parse_utterance('foo bar', verbose=True)
    assert foobar.log_chunkiness == log_chunkiness(foobar)
    assert numila.speak(['foo', 'bar']) == ['foo', 'bar']

def log_chunkiness(parse):

    def helper(chunk):
        if not hasattr(chunk, 'chunkiness'):
            return 0  # base case, this is a node.
        val = np.log(chunk.chunkiness())
        print('{chunk} log chunkiness = {val}'.format_map(locals()))
        return val + helper(chunk.child1) + helper(chunk.child2)

    return np.prod([helper(c) for c in parse]) # ** (1 / (len(parse.utterance) - 1))