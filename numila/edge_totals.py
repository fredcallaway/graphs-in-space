import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')

from vectorgraph import VectorGraph

def sim():
    graph = VectorGraph(DIM=500)
    ids = [str(i) for i in range(1000)]
    for c in ids:
        node = graph.create_node(c)
        graph.add(node)

    counts = {c: 0 for c in ids}
    samples = [(i, c) for i, c in enumerate(ids)
               if not i % 5]
    for i, c in samples:
        n1 = graph[c]
        for c2 in ids[i:]:
            n2 = graph[c2]
            n1.bump_edge(n2)
            counts[c] += 1

    for i, c in samples:
        n1 = graph[c]
        total_edge = sum(n1.edge_weight(n2)
                         for n2 in graph.nodes)
        yield {'number of edges': counts[c], 'total edge weight': total_edge}

def sim2():
    graph = VectorGraph(DIM=500)
    ids = [str(i) for i in range(10)]
    for c in ids:
        node = graph.create_node(c)
        graph.add(node)

    n1, n2, n3, n4 = map(graph.get, '1234')

    #for n in (n2, n3):
    n1.bump_edge(n2)
    print(n1.edge_weight(n2))
    exit()

    n2.bump_edge(n3)

    for x in (n1, n2):
        for y in (n2, n3, n4):
            print(x, '->', y, '=', x.edge_weight(y))

def main():
    df = pd.DataFrame(sim()).sort_values('number of edges')
    #df.plot('number of edges', 'total edge weight')
    sns.lmplot('number of edges', 'total edge weight', data=df, fit_reg=False)
    sns.plt.savefig('figs/total-edges.pdf')

if __name__ == '__main__':
    #main()
    sim2()