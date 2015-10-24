from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # is used implicitly
from mpl_toolkits.mplot3d import proj3d
from sklearn import cluster, manifold
from scipy.cluster import hierarchy
import numpy as np
import os
import pandas as pd
import seaborn as sns

labels_and_points = []

def mds(distance_matrix, interactive=False, dim=2, clustering=True, clusters=4):
    """Saves a scatterplot of the labels projected onto 2 dimensions.

    Uses MDS to project features onto a 2 or 3 dimensional based on their
    distances from each other. If features is not None, only plot those
    that are given. If interactive is truthy, an interactive plot will pop up. This is
    recommended for 3D graphs which are hard to make sense of without
    rotating the graph.
    """
    labels = distance_matrix.columns

    if clustering:
        clustering = cluster.AgglomerativeClustering(
                        linkage='complete', affinity='precomputed', n_clusters=clusters)
        assignments = clustering.fit_predict(distance_matrix)
    
    if dim == 2:
        mds = manifold.MDS(n_components=2, eps=1e-9, dissimilarity="precomputed")
        points = mds.fit(distance_matrix).embedding_

        plt.scatter(points[:,0], points[:,1], c=assignments, s=40)
        for label, x, y in zip(labels, points[:, 0], points[:, 1]):
            plt.annotate(label, xy = (x, y), xytext = (-5, 5),
                         textcoords = 'offset points', ha = 'right', va = 'bottom')
    else:
        if dim is not 3:
            raise ValueError('dim must be 2 or 3. {} provided'.format(dim))
        mds = manifold.MDS(n_components=3, eps=1e-9, dissimilarity="precomputed")
        points = mds.fit(distance_matrix).embedding_

        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        xs, ys, zs = np.split(points, 3, axis=1)
        ax.scatter(xs,ys,zs, c=assignments, s=40)

        # make labels move as the user rotates the graph
        global labels_and_points  # a hack for namespace problems
        labels_and_points = []
        for feature, x, y, z in zip(labels, xs, ys, zs):
            x2, y2, _ = proj3d.proj_transform(x,y,z, ax.get_proj())
            full_label = plt.annotate(
                feature, 
                xy = (x2, y2), xytext = (-5, 5),
                textcoords = 'offset points', ha = 'right', va = 'bottom',)
            labels_and_points.append((full_label, x, y, z))

        def update_position(e):
            for full_label, x, y, z in labels_and_points:
                x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
                full_label.xy = x2,y2
                full_label.update_positions(fig.canvas.renderer)
            fig.canvas.draw()

        fig.canvas.mpl_connect('motion_notify_event', update_position)

    os.makedirs('figs', exist_ok=True)
    plt.savefig('figs/mds{}.png'.format(dim))
    if interactive:
        plt.show()


def dendrogram(distance_matrix, method='complete'):
    """Plots a dendrogram using hierarchical clustering.

    see scipy.cluster.hierarchy.linkage for details regarding
    possible clustering methods.
    """
    labels = distance_matrix.columns
    clustering = hierarchy.linkage(distance_matrix, method=method)
    hierarchy.dendrogram(clustering, orientation='left', truncate_mode=None,
                                 labels=labels, color_threshold=0)
    plt.tight_layout()

    os.makedirs('figs', exist_ok=True)
    plt.savefig('figs/dendrogram2.png')
    plt.show()

def distance_matrix(distance_matrix):
    plt.figure(figsize=(13, 10))
    p = sns.heatmap(distance_matrix)
    p.set_xticklabels(distance_matrix.index, rotation=90)
    p.set_yticklabels(list(reversed(distance_matrix.index)), rotation=0)
    sns.plt.show()


    ## plotting the correlation matrix
    #R = np.corrcoef(distance_matrix)
    #import IPython; IPython.embed()
    #plt.pcolormesh(distance_matrix)
    #plt.colorbar()
    ##plt.yticks(arange(0.5,10.5),range(0,10))
    ##plt.xticks(arange(0.5,10.5),range(0,10))
    #plt.show()

if __name__ == '__main__':
    distance_matrix(pd.read_pickle('pcfg2-distances.pkl'))
