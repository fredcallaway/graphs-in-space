from graph import Graph, Node

def test_graph():
    graph = Graph(['edge'], {'DECAY_RATE': 0.1})
    n1 = Node(graph, 'n1')
    assert graph.get('n1') is None
    graph.add_node(n1)
    assert graph['n1'] is n1
    assert graph.get('n1') is n1

    n2 = Node(graph, 'n2')
    graph.add_node(n2)
    n1.bump_edge('edge', n2, 3)
    assert n1.edge_weight('edge', n2) == 1
    assert n2.edge_weight('edge', n1) == 0

    n3 = Node(graph, 'n3')
    graph.add_node(n3)
    n1.bump_edge('edge', n3, 1)
    assert n1.edge_weight('edge', n2) == 3 / 4
    assert n1.edge_weight('edge', n3) == 1 / 4

    graph.decay()
    assert n1.edge_weight('edge', n2) == (3 - 0.1) / (4 - 0.2)
    assert n1.edge_weight('edge', n3) == (1 - 0.1) / (4 - 0.2)
    
    for _ in range(10):
        graph.decay()

    assert n1.edge_weight('edge', n3) == 0
    assert n1.edge_weight('edge', n2) == 1




if __name__ == '__main__':
    test_graph()