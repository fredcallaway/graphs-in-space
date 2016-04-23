import numpy as np
import pandas as pd

from holograph import HoloGraph
from probgraph import ProbGraph
import utils




def sim():
    for dim in [100, 400, 1600, 6400]:
        for num_nodes in [50, 100, 200]:
            holo = HoloGraph(DIM=dim, INITIAL_ROW=0)
            prob = ProbGraph(DIM=dim)

            ids = [str(i) for i in range(num_nodes)]
            for id in ids:
                for graph in holo, prob:
                    node = graph.create_node(id)
                    graph.add(node)


            sources = utils.Bag({id: 100 for id in ids})
            sinks = utils.Bag({id: 100 for id in ids})

            while sources:
                id1 = sources.sample()
                id2 = sinks.sample()

                for graph in holo, prob:
                    n1 = graph[id1]
                    n2 = graph[id2]
                    n1.bump_edge(n2)

            holo_weights = [n1.edge_weight(n2)
                            for n1 in list(holo.nodes)[:25]
                            for n2 in holo.nodes]
            
            prob_weights = [n1.edge_weight(n2)
                            for n1 in list(prob.nodes)[:25]
                            for n2 in prob.nodes]
a
            #df = pd.DataFrame({'holo_weights': holo_weights,
                               #'prob_weights': prob_weights})
            for hw, pw in zip(holo_weights, prob_weights):
                yield {'dim': dim,
                       'nodes': num_nodes,
                       'holo_weights': hw,
                       'prob_weights': pw,
                       #'pearson': pearsonr(holo_weights, prob_weights)[0],
                       #'spearman': spearmanr(holo_weights, prob_weights)[0],
                       }


def main():
    #joblib.dump(list(sim()), 'pickles/compare_graphs', compress=3)
    df = pd.DataFrame(list(sim()))
    df.to_pickle('pickles/compare_graphs')
    print('wrote pickles/compare_graphs')

if __name__ == '__main__':
    main()

