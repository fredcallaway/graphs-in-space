from holograph import HoloGraph
from probgraph import ProbGraph
import pytest

@pytest.fixture
def holograph():
    graph = HoloGraph(['edge'], {'DIM': 1000, 'PERCENT_NON_ZERO': .01, 
                                'BIND_OPERATION': 'addition'})
    return graph

@pytest.fixture
def probgraph():
    return ProbGraph(['edge'], {'DECAY': 0.01})

@pytest.fixture(params=['holo', 'graph'])
def graph(request):
    if request.param == 'holo':
        graph = holograph()
    else:
        graph = probgraph()

    a = graph.create_node('A')
    b = graph.create_node('B')
    c = graph.create_node('C')
    assert 'A' not in graph

    graph.add_node(a)
    graph.add_node(b)
    graph.add_node(c)
    assert 'A' in graph

    return graph


def test_weights(graph):
    a, b, c = graph.nodes
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

if __name__ == '__main__':
    pytest.main([__file__])