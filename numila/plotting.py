from sklearn import manifold, cluster
import seaborn as sns
plt = sns.plt

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
