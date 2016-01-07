import numila
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

CORPUS = list(main.cfg_corpus(500))
Numila = numila.Numila
TEST = numila.TEST


def test_simple():
    model = Numila(DIM=1000, ADD_BOUNDARIES=False).fit(CORPUS)
    basic_tester(model)


def test_prob():
    model = Numila(DIM=1000, ADD_BOUNDARIES=False, GRAPH='probgraph').fit(CORPUS)
    basic_tester(model)


def test_generalization():
    model = Numila(DIM=100, GENERALIZE=('neighbor', 0.2), ADD_BOUNDARIES=False).fit(CORPUS)
    basic_tester(model)

    chunks = [n for n in model.graph.nodes if hasattr(n, 'chunkiness')]
    for chunk in chunks:
        pass

    TEST['on'] = True
    chunk = chunks[0]
    numila.neighbor_generalize(chunk, 0.5)
    print(chunk)
    print(TEST['similar_chunks'])
    


def verify_chunks(graph):
    chunks = [n for n in graph.nodes if hasattr(n, 'chunkiness')]

    # All chunk children are in the graph.
    for chunk in chunks:
        for child in (chunk.child1, chunk.child2):
            assert child is graph[child.id_string]

    # All chunks have unique ids
    unique_ids = set(c.id_string for c in chunks)
    assert len(unique_ids) == len(chunks)


def basic_tester(numila):
    verify_chunks(numila.graph)

    return

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

    return sum(helper(c) for c in parse) # ** (1 / (len(parse.utterance) - 1))