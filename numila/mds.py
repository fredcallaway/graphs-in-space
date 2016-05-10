from collections import Counter
import numpy as np
import pandas as pd
import sklearn
from sklearn import manifold, cluster
import scipy
import seaborn as sns
plt = sns.plt

from vectorgraph import VectorGraph
from probgraph import ProbGraph
import utils

def toy():
    dim = 1000
    num_nodes = 20

    vector = VectorGraph(DIM=dim)
    prob = ProbGraph(DIM=dim)

    ids = [str(i) for i in range(num_nodes)]
    for id in ids:
        for graph in vector, prob:
            node = graph.create_node(id)
            graph.add(node)


    a, b, c = map(graph.get, '123')
    x, y, z = map(graph.get, '456')

    a.bump_edge(x, factor=5)
    b.bump_edge(y, factor=5)
    
    c.bump_edge(x, factor=5)
    c.bump_edge(z, factor=1)


    #sources = utils.Bag({id: 100 for id in ids})
    #sinks = utils.Bag({id: 100 for id in ids})

    #while sources:
    #    id1 = sources.sample()
    #    id2 = sinks.sample()

    #    for graph in vector, prob:
    #        n1 = graph[id1]
    #        n2 = graph[id2]
    #        n1.bump_edge(n2)

    #mds(vector.nodes, name='vector_mds')
    #mds(prob.nodes, name='prob_mds')


from numila import Numila

def main():
    lang = 'English'
    kind = 'word'
    train_len = 4000
    N = 20
    
    model = Numila()
    corpus = utils.get_corpus(lang, kind)
    #import pcfg
    #corpus = (s.split(' ') for s in pcfg.toy2())

    train_corpus = [next(corpus) for _ in range(train_len)]
    model.fit(train_corpus)

    top_words, _ = zip(*Counter(utils.flatten(train_corpus)).most_common(N))

    nodes = [model.graph[w] for w in top_words]
    data = [[1-n1.similarity(n2) for n2 in nodes]
            for n1 in nodes]

    mds(data, top_words)

def mds(data, labels, clustering=False, dim=2, metric=True, n_clusters=2, name='mds'):

    assignments = []
    if clustering:
        clustering = cluster.AgglomerativeClustering(
                        linkage='complete', n_clusters=n_clusters)
        assignments = clustering.fit_predict(data)
    
    if dim == 2:
        mds = manifold.MDS(n_components=2, metric=metric, eps=1e-9, dissimilarity='precomputed')
        points = mds.fit(data).embedding_
        plt.scatter(points[:,0], points[:,1], s=40, c=assignments)  #  c=assignments
        for label, x, y in zip(labels, points[:, 0], points[:, 1]):
            plt.annotate(label, xy = (x, y), xytext = (-5, 5),
                         textcoords = 'offset points', ha = 'right', va = 'bottom')

    plt.savefig('figs/{}.pdf'.format(name))

if __name__ == '__main__':
    main()