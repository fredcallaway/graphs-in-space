from holograph import HoloGraph
from probgraph import ProbGraph
import pytest
import string
import vectors

@pytest.fixture
def holograph():
    graph = HoloGraph(['edge'], **{'DIM': 1000, 'PERCENT_NON_ZERO': .01, 'DYNAMIC': 0,
                                'BIND_OPERATION': 'addition', 'GENERALIZE': False},)
    _add_nodes(graph)
    return graph

@pytest.fixture
def probgraph():
    graph = ProbGraph(['edge'], **{'DECAY': 0.01, 'HIERARCHICAL': True})
    _add_nodes(graph)
    return graph

@pytest.fixture(params=['holo', 'graph'])
def graph(request):
    if request.param == 'holo':
        return holograph()
    else:
        return probgraph()

def _add_nodes(graph):
    for c in string.ascii_uppercase:
        node = graph.create_node(c)
        assert c not in graph
        graph.add(node)
        assert c in graph

    return graph

def test_weights(graph):
    a, b, c, d = (graph[x] for x in 'ABCD')
    edge_counts = [
       ((c, a), 8),
       ((b, a), 6),
       ((a, a), 4),
       # Reverse order because higher ->a edges means lower ->b edges
       ((a, b), 2),
       ((b, b), 2),
       ((c, b), 2),

       ((a, c), 1),
       ((b, c), 1),
       ((c, c), 1),
    ]

    for (n1, n2), count in edge_counts:
        n1.bump_edge(n2, 'edge', count)

    weights = [n1.edge_weight(n2, 'edge') for (n1, n2), _ in edge_counts]
    print(weights)
    assert sorted(weights, reverse=True) == weights

    #assert graph.edge_weight('edge', a, b) > graph.edge_weight('edge', b, c)
    assert a.edge_weight(b, 'edge') > b.edge_weight(c, 'edge')
    

def test_bind(holograph):
    graph = holograph
    graph.COMPOSITION = True
    a, b, c, d, e, f = (graph[x] for x in 'ABCDEF')
    
    # (A and B) and (C and D) are given similar edge profiles.
    a.bump_edge(c, 'edge', 5)
    a.bump_edge(d, 'edge', 3)
    b.bump_edge(c, 'edge', 3)
    b.bump_edge(d, 'edge', 5)
    assert a.similarity(b) > 0.5
    assert b.similarity(a) > 0.5
    c.bump_edge(a, 'edge', 5)
    c.bump_edge(b, 'edge', 3)
    d.bump_edge(a, 'edge', 3)
    d.bump_edge(b, 'edge', 5)
    assert c.similarity(d) > 0.5
    assert d.similarity(c) > 0.5

    # Create AC and give it an edge profile.
    ac = graph.bind(a, c)
    graph.add(ac)
    ac.bump_edge(e, 'edge', 5)
    ac.bump_edge(f, 'edge', 3)

    # BD should be similar to AC.
    bd = graph.bind(b, d)
    assert bd.similarity(ac) > 0.5
    assert bd.edge_weight(e, 'edge') > 0.5
    assert bd.edge_weight(f, 'edge') > 0.5
    assert bd.similarity(b) < 0.1
    assert bd.similarity(d) < 0.1

    # Creating link to ac should not result in a link to bd.
    e.bump_edge(ac, 'edge', 5)
    assert e.edge_weight(bd, 'edge') < 0.1

    # But it *should* result in a weak link to af.
    af = graph.bind(a, f)
    assert e.edge_weight(af, 'edge') > 0.5    # or should it?


def test_flat_bind(probgraph):
    #graph = ProbGraph(['edge'], {'DECAY': 0.01, 'HIERARCHICAL': False})
    graph = probgraph
    graph.HIERARCHICAL = False
    a, b, c, d, e, f = (graph[x] for x in 'ABCDEF')
    ab = graph.bind(a, b)
    cde = graph.bind(c, d, e)
    abcde = graph.bind(ab, cde)
    print(ab)
    print(cde)
    print(abcde)


def test_dynamic_generalize():
    graph = HoloGraph(['edge'], DIM=1000, PERCENT_NON_ZERO=.01, DYNAMIC=1)
    _add_nodes(graph)
    a, b, c, d, e, f = (graph[x] for x in 'ABCDEF')

    edge_counts = [
        ((a, c), 5),
        ((a, d), 5),
        ((a, e), 5),
        ((b, d), 5),
        ((b, e), 5),
        ((b, f), 5),
    ]

    for (n1, n2), count in edge_counts:
        n1.bump_edge(n2, 'edge', count)

    print('--- NORMAL WEIGHTS ---')
    print('a -> c', a.edge_weight(c, 'edge'))
    print('a -> d', a.edge_weight(d, 'edge'))
    print('a -> e', a.edge_weight(e, 'edge'))
    print('b -> d', b.edge_weight(d, 'edge'))
    print('b -> e', b.edge_weight(e, 'edge'))
    print('b -> f', b.edge_weight(f, 'edge'))
    print('b -> c', b.edge_weight(c, 'edge'))

    print('--- GENERALIZED WEIGHTS ---')
    print('a -> c', a.edge_weight(c, 'edge', dynamic=True))
    print('a -> d', a.edge_weight(d, 'edge', dynamic=True))
    print('a -> e', a.edge_weight(e, 'edge', dynamic=True))
    print('b -> d', b.edge_weight(d, 'edge', dynamic=True))
    print('b -> e', b.edge_weight(e, 'edge', dynamic=True))
    print('b -> f', b.edge_weight(f, 'edge', dynamic=True))
    print('b -> c', b.edge_weight(c, 'edge', dynamic=True))

    assert a.edge_weight(c, 'edge') > 0.3
    assert a.edge_weight(d, 'edge') > 0.3
    assert a.edge_weight(e, 'edge') > 0.3
    assert b.edge_weight(d, 'edge') > 0.3
    assert b.edge_weight(e, 'edge') > 0.3
    assert b.edge_weight(f, 'edge') > 0.3

    # A links to C, so C's dynamic vec should link to A.
    assert vectors.cosine(c.dynamic_row_vecs['edge'], a.row_vecs['edge']) > 0.4

    # B is connected to C because A is connected to C
    # and B is connected to similar nodes as A.
    assert b.edge_weight(c, 'edge', dynamic=True) > 0.2


def test_full_generalize(holograph):
    a, b, c, d, e, f = (holograph[x] for x in 'ABCDEF')

    edge_counts = [
        ((a, c), 5),
        ((a, d), 5),
        ((a, e), 5),
        ((b, d), 5),
        ((b, e), 5),
        ((b, f), 5),
    ]

    for (n1, n2), count in edge_counts:
        n1.bump_edge(n2, 'edge', count)

    print('--- NORMAL WEIGHTS ---')
    print('a -> c', a.edge_weight(c, 'edge'))
    print('a -> d', a.edge_weight(d, 'edge'))
    print('a -> e', a.edge_weight(e, 'edge'))
    print('a -> f', a.edge_weight(f, 'edge'))
    print('b -> c', b.edge_weight(c, 'edge'))
    print('b -> d', b.edge_weight(d, 'edge'))
    print('b -> e', b.edge_weight(e, 'edge'))
    print('b -> f', b.edge_weight(f, 'edge'))

    print('--- GENERALIZED WEIGHTS ---')
    gen = 1
    print('a -> c', a.edge_weight(c, 'edge', generalize=gen))
    print('a -> d', a.edge_weight(d, 'edge', generalize=gen))
    print('a -> e', a.edge_weight(e, 'edge', generalize=gen))
    print('a -> f', a.edge_weight(f, 'edge', generalize=gen))
    print('b -> c', b.edge_weight(c, 'edge', generalize=gen))
    print('b -> d', b.edge_weight(d, 'edge', generalize=gen))
    print('b -> e', b.edge_weight(e, 'edge', generalize=gen))
    print('b -> f', b.edge_weight(f, 'edge', generalize=gen))

    assert a.edge_weight(c, 'edge') > 0.3
    assert a.edge_weight(d, 'edge') > 0.3
    assert a.edge_weight(e, 'edge') > 0.3
    assert b.edge_weight(d, 'edge') > 0.3
    assert b.edge_weight(e, 'edge') > 0.3
    assert b.edge_weight(f, 'edge') > 0.3


    # B is connected to C only by generalization
    assert b.edge_weight(c, 'edge') < 0.1
    assert b.edge_weight(c, 'edge', generalize=gen) > 0.2
    
    assert a.edge_weight(f, 'edge') < 0.1
    assert a.edge_weight(f, 'edge', generalize=gen) > 0.2
    

if __name__ == '__main__':
    pytest.main([__file__])