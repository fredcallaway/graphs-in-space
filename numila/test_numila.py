import numila
import numpy as np
import utils
import pytest

Numila = numila.Numila



def test_parse():
    model = Numila(GRAPH='probgraph', LEARNING_RATE=1, EXEMPLAR_THRESHOLD=1,
                   DECAY=0)

    def log_parse(utt):
        with utils.capture_logging('parse') as log:
            model.parse(utt)
        log = log()
        print(log)
        return log

    log = log_parse('a b c')
    assert log.count('strengthen a -> b') is 2
    assert log.count('strengthen b -> c') is 2

    log = log_parse('a b c d')
    assert log.count('strengthen a -> b') is 3
    assert log.count('strengthen b -> c') is 3
    assert log.count('strengthen c -> d') is 3

    log = log_parse('a b c d e')
    assert log.count('strengthen a -> b') is 3
    assert log.count('strengthen b -> c') is 3
    assert log.count('strengthen c -> d') is 3
    assert log.count('strengthen d -> e') is 3



def test_holo():
    return
    model = Numila(GRAPH='holograph', LEARNING_RATE=0.1, EXEMPLAR_THRESHOLD=1)
    _test_toy(model)

def test_prob():
    return
    model = Numila(GRAPH='probgraph', LEARNING_RATE=1, EXEMPLAR_THRESHOLD=1,
                   DECAY=0)
    _test_toy(model)


def _test_toy(model):
    # One simple utterance 50 times.
    corpus = ['a b a c b d'] * 50
    model.parse(corpus[0])
    a, b, c, d = (model.graph[x] for x in 'abcd')  # node objects
    weight = model.graph.edge_weight  # shorten function name
    def weight(edge, n1, n2):
        return n1.edge_weight(n2, edge)

    # Check that all connections are positive after one utterance
    assert weight('ftp', a, b)
    assert weight('ftp', b, a)
    assert weight('ftp', a, c)
    assert weight('ftp', c, a)

    assert weight('btp', b, a)
    assert weight('btp', a, b)
    assert weight('btp', c, a)
    assert weight('btp', a, c)

    # Equal conditional probability, but more evidence
    assert weight('btp', b, a) > weight('btp', c, a)

    model.fit(corpus)

    # Check the connections have stayed positive after 50 more.
    assert weight('ftp', a, b)
    assert weight('ftp', b, a)
    assert weight('ftp', a, c)
    assert weight('ftp', c, a)

    assert weight('btp', b, a)
    assert weight('btp', a, b)
    assert weight('btp', c, a)
    assert weight('btp', a, c)

    w1 = weight('ftp', a, b)
    model.parse('b c')
    w2 = weight('ftp', a, b)
    assert w1 - w2 < .001

    w1 = weight('btp', b, a)
    model.parse('d a d a d a d a d a d a')
    w2 = weight('btp', b, a)
    assert w1 - w2 < .001

    
    # Check that more common edges are more highly weighted.
    # We vary the conditional (ab | a) and raw (ab) probabilities.
    # Reference: a b a c a b d

    # Higher conditional, higher raw.
    assert weight('ftp', a, b) > weight('ftp', a, c)
    
    # Higher conditional, equal raw.
    assert weight('ftp', c, a) > weight('ftp', b, d)

    # Equal conditional, higher raw. But lots of evidence for both.
    print()
    print(weight('btp', c, a, verbose=True))
    assert 0

    assert weight('btp', b, a) - weight('btp', c, a) < 0.001

    
    # This always fails for holo. The edge weights do not really
    # represent probabilities. They are more sensitive to the raw
    # occurrence counts.
    # p(ab | a) = 0.66
    # p(ca | c) = 1
    # p(ab) = 0.4
    # p(ca) = 0.2
    #assert weight('ftp', c, a) > weight('ftp', a, b)

    assert weight('ftp', a, a) < 0.05
    assert weight('ftp', b, b) < 0.05
    assert weight('ftp', c, c) < 0.05
    assert weight('ftp', b, c) < 0.05

    #assert ''.join(model.speak('caab')) == 'abac'
    #assert ''.join(model.speak('cab')) in ('bac', 'cab')

if __name__ == '__main__':
    pytest.main(__file__)
    #test_prob()

"""

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

    parse = numila.parse('the boy ate')
    
    equal = parse.log_chunkiness == log_chunkiness(parse)
    # We check explicitly for np.nan because np.nan != np.nan
    both_nan = np.isnan(parse.log_chunkiness) or np.isnan(log_chunkiness(parse))
    assert equal or both_nan

    assert not np.isnan(parse.log_chunkiness)

    print('\nFIRST parse of foo bar')
    numila.parse('foo bar', verbose=True)
    #numila.parse('foo bar')
    #numila.parse('foo bar')
    #numila.parse('foo bar')
    print('\nSECOND parse of foo bar')
    foobar = numila.parse('foo bar', verbose=True)
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

"""