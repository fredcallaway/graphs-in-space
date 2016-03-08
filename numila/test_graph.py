from holograph import HoloGraph
from probgraph import ProbGraph
import pytest
import string
import vectors

@pytest.fixture
def holograph():
    graph = HoloGraph(['edge'], {'DIM': 1000, 'PERCENT_NON_ZERO': .01, 
                                'BIND_OPERATION': 'addition', 'GENERALIZE': False})
    _add_nodes(graph)
    return graph

@pytest.fixture
def probgraph():
    graph = ProbGraph(['edge'], {'DECAY': 0.01})
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
        graph.add_node(node)
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
    graph.params['COMPOSITION'] = True
    a, b, c, d, e, f = (graph[x] for x in 'ABCDEF')
    
    # (A and B) and (C and D) are given similar edge profiles.
    a.bump_edge(c, 'edge', 5)
    a.bump_edge(d, 'edge', 3)
    b.bump_edge(c, 'edge', 3)
    b.bump_edge(d, 'edge', 5)
    assert a.similarity(b) > 0.5
    c.bump_edge(a, 'edge', 5)
    c.bump_edge(b, 'edge', 3)
    d.bump_edge(a, 'edge', 3)
    d.bump_edge(b, 'edge', 5)
    assert b.similarity(a) > 0.5

    # Create AC and give it an edge profile.
    ac = graph.bind(a, c)
    graph.add_node(ac)
    ac.bump_edge(e, 'edge', 5)
    ac.bump_edge(f, 'edge', 3)

    # BD should be similar to AC.
    bd = graph.bind(b, d)
    assert bd.similarity(ac) > 0.5
    assert bd.similarity(b) < 0.1
    assert bd.similarity(d) < 0.1

    # Creating link to ac should not result in a link to bd.
    e.bump_edge(ac, 'edge', 5)
    assert e.edge_weight(bd, 'edge') < 0.1

    # But it *should* result in a weak link to af.
    af = graph.bind(a, f)
    #assert e.edge_weight(af, 'edge') > 0.5    # or should it?

def test_dynamic_gen(holograph):
    graph = HoloGraph(['edge'], {'DIM': 1000, 'PERCENT_NON_ZERO': .01, 
                                'BIND_OPERATION': 'addition', 'GENERALIZE': 'dynamic2',
                                'DYNAMIC':0.3})
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

    assert a.edge_weight(c, 'edge') > 0.3
    assert a.edge_weight(d, 'edge') > 0.3
    assert a.edge_weight(e, 'edge') > 0.3
    assert b.edge_weight(d, 'edge') > 0.3
    assert b.edge_weight(e, 'edge') > 0.3
    assert b.edge_weight(f, 'edge') > 0.2
    
    assert vectors.cosine(d.dynamic_vec, a.row_vec) > 0.4
    # B is connected to C because A is connected to C
    # and B is connected to similar nodes as A.
    assert b.edge_weight(c, 'edge') > 0.2






if __name__ == '__main__':
    pytest.main([__file__])