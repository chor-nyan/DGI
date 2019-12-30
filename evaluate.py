import networkx as nx
import numpy as np
import umap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from munkres import Munkres
import csv
from numpy import savetxt
from pandas import DataFrame
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, f1_score
import os
import glob
from matplotlib.backends.backend_pdf import PdfPages
# from MantelTest import Mantel
from hub_toolbox.distances import euclidean_distance
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numba
from MantelTest import Mantel
# From: https://github.com/leowyy/GraphTSNE/blob/master/util/evaluation_metrics.py

import numpy as np
from sklearn import neighbors
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from scipy.sparse.csgraph import shortest_path
import time

# # 隣接行列A
# A=np.matrix([[1,1],[2,1]])
# G=nx.from_numpy_matrix(A)
#
# # A = np.squeeze(np.asarray(A))
#
# # 特徴行列Xがnode_pos
# node_pos = X
#
# def plot_embedding2D(node_pos, node_colors=None, di_graph=None):
#     node_num, embedding_dimension = node_pos.shape
#     if(embedding_dimension > 2):
#         print("Embedding dimension greater than 2, use tSNE to reduce it to 2")
#         model = umap.UMAP(n_components=2)
#         node_pos = model.fit_transform(node_pos)
#
#     if di_graph is None:
#         # plot using plt scatter
#         plt.scatter(node_pos[:, 0], node_pos[:, 1], c=node_colors)
#     else:
#         # plot using networkx with edge structure
#         pos = {}
#         for i in range(node_num):
#             pos[i] = node_pos[i, :]
#         if node_colors:
#             nx.draw_networkx_nodes(di_graph, pos,
#                                    node_color=node_colors,
#                                    width=0.1, node_size=100,
#                                    arrows=False, alpha=0.8,
#                                    font_size=5)
#         else:
#             nx.draw_networkx(di_graph, pos, node_color=node_colors,
#                              width=0.1, node_size=300, arrows=False,
#                              alpha=0.8, font_size=12)
#
# plot_embedding2D(A, di_graph=G)

# Input: emb_features-X, label-L, adjMatrix-A

def plot_graph(X, L, A):
    node_pos = X
    node_num, embedding_dimension = node_pos.shape

    if not isinstance(A, np.ndarray):
        A = A.todense()

    node_colors = L.astype(int)
    di_graph = nx.from_numpy_matrix(A)
    # plot using networkx with edge structure
    pos = {}
    for i in range(node_num):
        pos[i] = node_pos[i, :]
    if node_colors.any():
        nx.draw_networkx(di_graph, pos,
                         node_color=node_colors,
                         width=0.1, node_size=10,
                         arrows=False, alpha=0.8,
                         cmap='Spectral', with_labels=False)
    else:
        nx.draw_networkx(di_graph, pos, node_color=node_colors,
                         width=0.1, node_size=1, arrows=False,
                         alpha=0.8, with_labels=False, cmap='Spectral')

def kNN_acc(X, L):
    X_train, X_test, Y_train, Y_test = train_test_split(X, L, random_state=0)
    knc = KNeighborsClassifier(n_neighbors=1)
    knc.fit(X_train, Y_train)
    Y_pred = knc.predict(X_test)
    score = knc.score(X_test, Y_test)

    return score



def save_visualization(X, L, A, dir='./fig_vis/', dataset = 'cora', DGI = True, i=0):

    sns.set(context="paper", style="white")
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.setp(ax, xticks=[], yticks=[])
    plot_graph(X, L, A)
    model = 'DGI-UMAP' if DGI else "UMAP"
    plt.savefig(dir + dataset + '_' + model + str(i+1) + '.png')


def kmeans_acc_ari_ami_f1(X, L, verbose=1):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    n_clusters = len(np.unique(L))
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)

    y_pred = kmeans.fit_predict(X)
    y_pred = y_pred.astype(np.int64)
    y_true = L.astype(np.int64)
    assert y_pred.size == y_true.size

    y_pred = y_pred.reshape((1, -1))
    y_true = y_true.reshape((1, -1))

    # D = max(y_pred.max(), L.max()) + 1
    # w = np.zeros((D, D), dtype=np.int64)
    # for i in range(y_pred.size):
    #     w[y_pred[i], L[i]] += 1
    # # from sklearn.utils.linear_assignment_ import linear_assignment
    # from scipy.optimize import linear_sum_assignment
    # row_ind, col_ind = linear_sum_assignment(w.max() - w)
    #
    # return sum([w[i, j] for i in row_ind for j in col_ind]) * 1.0 / y_pred.size

    if len(np.unique(y_pred)) == len(np.unique(y_true)):
        C = len(np.unique(y_true))

        cost_m = np.zeros((C, C), dtype=float)
        for i in np.arange(0, C):
            a = np.where(y_pred == i)
            # print(a.shape)
            a = a[1]
            l = len(a)
            for j in np.arange(0, C):
                yj = np.ones((1, l)).reshape(1, l)
                yj = j * yj
                cost_m[i, j] = np.count_nonzero(yj - y_true[0, a])

        mk = Munkres()
        best_map = mk.compute(cost_m)

        (_, h) = y_pred.shape
        for i in np.arange(0, h):
            c = y_pred[0, i]
            v = best_map[c]
            v = v[1]
            y_pred[0, i] = v

        acc = 1 - (np.count_nonzero(y_pred - y_true) / h)

    else:
        acc = 0
    # print(y_pred.shape)
    y_pred = y_pred[0]
    y_true = y_true[0]
    ari, ami = adjusted_rand_score(y_true, y_pred), adjusted_mutual_info_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    results = {}
    results["ACC"] = acc
    results["ARI"] = ari
    results['AMI'] = ami
    results['F1_score'] = f1
    if verbose:
        for k, v in results.items():
            print("{} = {:.4f}".format(k, v))

    return acc, ari, ami, f1

@numba.jit()
def mantel_test(X, L, embed, verbose=1):
    sss = StratifiedShuffleSplit(n_splits=50, test_size=1000, random_state=0)
    sss.get_n_splits(X, L)

    label_type = list(set(L))
    r_lst = np.array([])
    p_lst = np.array([])
    for _, idx in sss.split(X, L):
        # print('Index: ', idx)
        # X_test = X[idx]
        # y_train =

        X_high, L_hl = X[idx], L[idx]
        X_low = embed[idx]

        # print(X_high.shape, L_high.shape)
        # print(X_low.shape, L_low.shape)

        label_idx = []

        for _, i in enumerate(label_type):
            l_idx = np.where(L_hl == i)
            label_idx.append(l_idx)

        # print(label_type)

        # label_idx
        X_high_lst = []
        X_low_lst = []
        # for _, i in enumerate(label_type):
        #     X_high_lst.append(X_high[label_idx[i]])
        for i, _ in enumerate(label_type):
            centroid = np.mean(X_high[label_idx[i]], axis=0)
            # print(centroid)
            X_high_lst.append(centroid)
            # print(centroid.shape)
            # X_high_lst.append((X_high[label_idx[i]] - centroid))
            # X_high_lst[label_idx[i]] = np.sqrt(np.linalg.norm(X_high[label_idx[i]] - centroid, ord=2))
            # for _, i in enumerate(label_type):

            centroid = np.mean(X_low[label_idx[i]], axis=0)
            X_low_lst.append(centroid)
            # print(centroid.shape)
            # X_high_lst.append((X_low[label_idx[i]] - centroid))
            # X_low_lst[label_idx[i]] = np.sqrt(np.linalg.norm(X_low[label_idx[i]] - centroid, ord=2))

        # print(X_low_lst[0].shape, centroid.shape)
        D_high = euclidean_distance(X_high_lst)
        D_low = euclidean_distance(X_low_lst)
    # print(D_high, D_low)

        r, p, z = Mantel.test(D_high, D_low, perms=10000, method='pearson', tail='upper')
        r_lst = np.append(r_lst, r)
        p_lst = np.append(p_lst, p)

    if verbose:
        print(p_lst)
        print(pd.DataFrame(pd.Series(r_lst.ravel()).describe()).transpose())

    return r_lst, p_lst
    # # return np.mean(r_lst)
    # print(pd.DataFrame(pd.Series(r_lst.ravel()).describe()).transpose())
    # print('r: ', r, 'p: ', p, 'z: ', z)

def box_plot_PCC(r_lst_org, r_lst_hub, save=False, dir='./fig_boxplot/', dataset = 'F-MNIST', i=0):

    # sns.set()
    sns.set(context="paper")
    # colors = ['blue', 'red']
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # ax.set_title('medianprops')
    # medi_style = dict(color='b', lw=30)

    ax.boxplot([r_lst_org, r_lst_hub], patch_artist=True, labels=['UMAP', 'HR-UMAP'])

    # for b, c in zip(bp['boxes'], colors):
    #     b.set(color=c, linewidth=1)  # boxの外枠の色
    #     b.set_facecolor(c)  # boxの色

    ax.set_xlabel('Model')
    ax.set_ylabel('Pearson correlation')
    ax.set_ylim(0.2, 0.8)

    if save:
        plt.savefig(dir + dataset + '_boxplot_' + str(i+1) + '.png')
    else:
        plt.show()


def get_shortest_path_matrix(adj, verbose=0):
    """
    Computes the all-pairs shortest path matrix for an undirected and unweighted graph
    If a pair of nodes are not connected, places an arbitrarily large value
    """
    if verbose:
        print("Computing all pairs shortest path lengths for {} nodes...".format(adj.shape[0]))
    t_start = time.time()

    path_lengths_matrix = shortest_path(adj, directed=False, unweighted=True)
    path_lengths_matrix = np.array(path_lengths_matrix)
    flag = path_lengths_matrix == np.inf
    path_lengths_matrix[flag] = 1e6

    t_elapsed = time.time() - t_start
    if verbose:
        print("Time to compute shortest paths (s) = {:.4f}".format(t_elapsed))
    return path_lengths_matrix


def nearest_neighbours_generalisation_accuracy(X, y, n_neighbors=1, verbose=1):
    """
    Returns the average 10-fold validation accuracy of a NN classifier trained on the given embeddings
    Args:
        X (np.array): feature matrix of size n x d
        y (np.array): label matrix of size n x 1
        n_neighbors (int): number of nearest neighbors to be used for inference
    Returns:
        score (float): Accuracy of the NN classifier
    """
    kf = KFold(n_splits=10)
    kf.get_n_splits(X)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = neighbors.KNeighborsClassifier(n_neighbors)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))

    if verbose:
        print(np.average(scores))
    return np.average(scores)


def evaluate_viz_metrics(y_emb, X, L, A_sparse, verbose=1):
    """
    Returns the evaluation metrics given a low-dimensional map and a dataset
    Args:
         y_emb (np.array): low dimensional map of data points, matrix of size n x 2
         dataset (EmbeddingDataSet or GraphDataBlock): contains dataset info for evaluation purposes
         辞書型で与える; data = dataset['X'], L = dataset['L'], A = dataset['A'] <- SPARSE MATRIX
    Returns:
        results (dict): name (str) -> score (float)
    """
    results = {}

    path_matrix = get_shortest_path_matrix(A_sparse, verbose=1)

    results["Graph_trust"] = graph_trustworthiness(y_emb, path_matrix, n_neighbors=2)
    results["Feature_trust"] = trustworthiness(X, y_emb, n_neighbors=12)
    results["One NN accuracy"] = nearest_neighbours_generalisation_accuracy(y_emb, L, 1)
    results["Avg graph distance"], results["Avg feature distance"] = combined_dist_metric(y_emb, X, A_sparse, k=5)
    results['Total distance'] = results["Avg graph distance"] + results["Avg feature distance"]

    if verbose:
        for k, v in results.items():
            print("{} = {:.4f}".format(k, v))
    return results


def neighborhood_preservation(y_emb, path_matrix, max_graph_dist=2):
    """
    Returns the graph neighborhood preservation measure, between [0, 1]
    Args:
        y_emb (np.array): low dimensional map of data points, matrix of size n x 2
        path_matrix (np.array): all-pairs shortest path length matrix of size n x n
        max_graph_dist (int): value of r, which defines the r-hop neighborhood from each node
    Returns:
        score (float): graph neighborhood preservation measure
    """
    dist_X_emb = pairwise_distances(y_emb, squared=True)
    ind_X_emb = np.argsort(dist_X_emb, axis=1)[:, 1:]

    n_samples = y_emb.shape[0]
    t = 0.0
    for i in range(n_samples):
        # Find the r-hop neighborhood in the graph space
        graph_n = {k for k, v in enumerate(path_matrix[i]) if 0 < v <= max_graph_dist}
        if len(graph_n) == 0:
            t += 1
            continue

        # Find the k nearest neighborhood in the embedding space
        layout_n = set(ind_X_emb[i][:len(graph_n)])
        intersection_size = len(graph_n.intersection(layout_n))

        # Compute the Jaccard similarity
        t += intersection_size / (2*len(graph_n) - intersection_size)
    return t/n_samples


def combined_dist_metric(y_emb, feature_matrix, W, k=5):
    """
    Returns the visualization distance-based metrics
    Args:
        y_emb (np.array): low dimensional map of data points, matrix of size n x 2
        feature_matrix (np.array): high dimensional map of data points, matrix of size n x f
        W (scipy sparse matrix): adjacency matrix
        k (int): number of nearest neighbors to construct kNN graph K
    Returns:
        graph_dist (float): average distances in map computed based on graph structure
        feature_dist (float): average distances in map computed based on kNN graph in the feature space
    """
    # Standard normalization on the low dimensional datapoints
    scaler = StandardScaler()
    z_emb = scaler.fit_transform(y_emb)
    z_dist_matrix = pairwise_distances(z_emb, squared=True)

    # feature_dist_matrix = pairwise_distances(feature_matrix, metric='cosine')
    knn_graph = kneighbors_graph(feature_matrix, n_neighbors=k, mode='connectivity', metric='cosine',
                                 include_self=False)

    # Average edge length in the original graph
    graph_dist = np.sum(W.toarray() * z_dist_matrix) / W.getnnz()

    # Average edge length in the kNN graph
    feature_dist = np.sum(knn_graph.toarray() * z_dist_matrix) / knn_graph.getnnz()

    return graph_dist, feature_dist


def run_regression(train_embeds, train_labels, test_embeds, test_labels):
    """
    Measures the quality of embeddings as input to a SGD classifier
    """
    np.random.seed(1)
    from sklearn.linear_model import SGDClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import f1_score
    dummy = DummyClassifier()
    dummy.fit(train_embeds, train_labels)
    log = SGDClassifier(loss="log", n_jobs=10, tol=1e-3)
    log.fit(train_embeds, train_labels)
    print("F1 score:", f1_score(test_labels, log.predict(test_embeds), average="micro"))
    print("Random baseline f1 score:", f1_score(test_labels, dummy.predict(test_embeds), average="micro"))


def graph_trustworthiness(y_emb, path_matrix, n_neighbors=5):
    """
    Deprecated in favor of neighborhood preservation: Measures trustworthiness on graphs
    """
    dist_X_emb = pairwise_distances(y_emb, squared=True)
    ind_X_emb = np.argsort(dist_X_emb, axis=1)[:, 1:n_neighbors + 1]

    n_samples = y_emb.shape[0]
    t = 0.0
    min_sum = 0.0
    max_sum = 0.0
    for i in range(n_samples):
        ranks = path_matrix[i][ind_X_emb[i, :]]
        t += np.sum(ranks)
        lengths_from_i = sorted(path_matrix[i])
        min_sum += sum(lengths_from_i[1:n_neighbors + 1])
        max_sum += sum(lengths_from_i[-n_neighbors:])
    t = 1.0 - (t - min_sum) / (max_sum - min_sum)
    # t = max_sum - t / (max_sum - min_sum)

    return t


def trustworthiness(X, X_embedded, n_neighbors=5, metric='euclidean', precomputed=False):
    """Expresses to what extent the local structure is retained.
    The trustworthiness is within [0, 1]. It is defined as
    .. math::
        T(k) = 1 - \frac{2}{nk (2n - 3k - 1)} \sum^n_{i=1}
            \sum_{j \in U^{(k)}_i} (r(i, j) - k)
    where :math:`r(i, j)` is the rank of the embedded datapoint j
    according to the pairwise distances between the embedded datapoints,
    :math:`U^{(k)}_i` is the set of points that are in the k nearest
    neighbors in the embedded space but not in the original space.
    * "Neighborhood Preservation in Nonlinear Projection Methods: An
      Experimental Study"
      J. Venna, S. Kaski
    * "Learning a Parametric Embedding by Preserving Local Structure"
      L.J.P. van der Maaten
    Parameters
    ----------
    X : array, shape (n_samples, n_features) or (n_samples, n_samples)
        If the metric is 'precomputed' X must be a square distance
        matrix. Otherwise it contains a sample per row.
    X_embedded : array, shape (n_samples, n_components)
        Embedding of the training data in low-dimensional space.
    n_neighbors : int, optional (default: 5)
        Number of neighbors k that will be considered.
    precomputed : bool, optional (default: False)
        Set this flag if X is a precomputed square distance matrix.
    Returns
    -------
    trustworthiness : float
        Trustworthiness of the low-dimensional embedding.
    """
    if precomputed:
        dist_X = X
    elif metric == 'cosine':
        dist_X = pairwise_distances(X, metric='cosine')
    else:
        dist_X = pairwise_distances(X, squared=True)
    dist_X_embedded = pairwise_distances(X_embedded, squared=True)
    ind_X = np.argsort(dist_X, axis=1)
    ind_X_embedded = np.argsort(dist_X_embedded, axis=1)[:, 1:n_neighbors + 1]

    n_samples = X.shape[0]
    t = 0.0
    ranks = np.zeros(n_neighbors)
    for i in range(n_samples):
        for j in range(n_neighbors):
            ranks[j] = np.where(ind_X[i] == ind_X_embedded[i, j])[0][0]
        ranks -= n_neighbors
        t += np.sum(ranks[ranks > 0])
    t = 1.0 - t * (2.0 / (n_samples * n_neighbors *
                          (2.0 * n_samples - 3.0 * n_neighbors - 1.0)))
    return t

data = 'cora'
# data = 'citeseer'
# data = 'pubmed'

DGI = True
model = 'DGI-UMAP' if DGI else "UMAP"

iter = 10
# iter1 = str(4)
# iter2 = str(6)
file_path0 = "GraphTSNE/cora_data_graphtsne.npz"
# file_path = 'GraphTSNE/embed_' + data + '_' + model + "_" + str(iter) + '.npz'
file_path = 'GraphTSNE/embed_GraphTSNE_12_31_4_48.npz'
file_path1 = "GraphTSNE/embed_GraphTSNE_cora12_31_7_12.npz"
# file_path1 = 'embed_' + data + '_' + model + '_' + iter2 + 'th.npz'
# file_path2 = 'embed_' + data + '_' + model + '_' + iter1 + 'th.npz'
# file_path = "embed_org_NORB48600_Seed:42.npz"
# file_path = 'embed_' + hub_org + "_coil100" + str(7200) + '_' + str(10) + '.npz'

npz0 = np.load(file_path0)
npz = np.load(file_path)
npz1 = np.load(file_path1)
# npz2 = np.load(file_path2)
X = npz0['X']
L = npz0['L']
A = npz0['A']
emb0 = npz['emb']
emb1 = npz1['emb']
# emb2 = npz2['emb']
emb = np.vstack((emb0, emb1))
np.savez('embed_' + data + '_' + 'GraphTSNE' + '_' + str(10), X=X, L=L, emb=emb, A=A)


result_knn = []
result_acc = []
result_ari = []
result_ami = []
result_f1 = []

for i, e in enumerate(emb):

    knn_acc = nearest_neighbours_generalisation_accuracy(e, L)
    acc, ari, ami, f1 = kmeans_acc_ari_ami_f1(e, L)
    # save_visualization(e, L, A, dataset=data, DGI=True, i=i)
    # visualize(e, L)
    result_knn.append(knn_acc)
    result_acc.append(acc)
    result_ari.append(ari)
    result_ami.append(ami)
    result_f1.append(f1)

print(result_knn, result_acc, result_ari, result_ami, result_f1)

iter = str(10)

# LOCAL accuracy ==========================
result = np.array((result_knn, result_acc, result_ari, result_ami, result_f1))
# with open('examples/result_org_'+data+datasize+'.csv', 'w') as f:
np.savetxt('embed_' + data + '_' + model + iter + '.txt', result)

# 統計処理
file_path = 'embed_' + data + '_' + model + iter + '.txt'
result_lst = np.loadtxt(file_path)
results = DataFrame()
results['knn'] = result_lst[0]
results['acc'] = result_lst[1]
results['ari'] = result_lst[2]
results['ami'] = result_lst[3]
results["f1"] = result_lst[4]
# descriptive stats
print(results.describe())