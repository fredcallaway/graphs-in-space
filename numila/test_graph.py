from holograph import HoloGraph
from probgraph import ProbGraph
import pytest

@pytest.fixture
def holograph():
    return HoloGraph(['edge'], {'DIM': 1000, 'PERCENT_NON_ZERO': .01, 
                                'BIND_OPERATION': 'addition'})
@pytest.fixture
def probgraph():
    return ProbGraph(['edge'], {'DECAY_RATE': 0.01})

def test_probgraph(probgraph):
    _test_graph(probgraph)

def test_holograph(holograph):
    _test_graph(holograph)

def _test_graph(graph):
    a = graph.create_node('A')
    b = graph.create_node('B')
    c = graph.create_node('C')
    assert 'A' not in graph

    graph.add_node(a)
    graph.add_node(b)
    graph.add_node(c)
    assert 'A' in graph
    
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

    for pair, count in edge_counts:
        graph.bump_edge('edge', *pair, count)

    weights = [graph.edge_weight('edge', *pair) for pair, _ in edge_counts]
    print(weights)
    assert sorted(weights, reverse=True) == weights


    assert graph.edge_weight('edge', a, b) > graph.edge_weight('edge', b, c)


#def test_equivalence(holograph, probgraph):
#    holo, prob = holograph, probgraph

#    for graph in (holo, prob):
#        a = graph.create_node('A')
#        b = graph.create_node('B')
#        c = graph.create_node('C')

#        graph.add_node(a)
#        graph.add_node(b)
#        graph.add_node(c)



if __name__ == '__main__':
    pytest.main([__file__])