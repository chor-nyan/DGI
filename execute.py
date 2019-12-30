import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from models import DGI, LogReg
from utils import process

from openTSNE import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import random
import networkx as nx

from evaluation_metrics import nearest_neighbours_generalisation_accuracy, evaluate_viz_metrics
from evaluate import kmeans_acc_ari_ami_f1, mantel_test

import time

UMAP_time_lst = []
DGI_time_lst = []

emb_UMAP_lst = []
emb_DGIUMAP_lst = []

iter_n = 1
seed_lst = random.sample(range(100), k=iter_n)
print(seed_lst)

dataset = 'cora'
# dataset = 'citeseer'
# dataset = 'pubmed'

for iter_number in range(iter_n):

    seed = seed_lst[iter_number]

    # training params
    batch_size = 1
    nb_epochs = 1
    patience = 20
    lr = 0.001
    l2_coef = 0.0
    drop_prob = 0.0
    hid_units = 256 if dataset == 'pubmed' else 512

    sparse = True
    nonlinearity = 'prelu'  # special name to separate parameters

    adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
    features, _ = process.preprocess_features(features)

    print(adj.shape, features.shape)
    A = adj
    A = A.todense()

    nolabel = []
    for i, r in enumerate(labels):
        if all(r == 0):
            nolabel.append(i)
    print(nolabel)

    # L = labels
    L = []
    for i, r in enumerate(labels):
        if i in nolabel:
            L.append(labels.shape[1])
        else:
            L.append(np.where(r == 1)[0][0])

    # L = np.array([np.where(r == 1)[0][0] for r in labels])
    L = np.array(L)
    print(L)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = labels.shape[1]

    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

    if sparse:
        sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    else:
        adj = (adj + sp.eye(adj.shape[0])).todense()

    features = torch.FloatTensor(features[np.newaxis])
    if not sparse:
        adj = torch.FloatTensor(adj[np.newaxis])
    labels = torch.FloatTensor(labels[np.newaxis])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    model = DGI(ft_size, hid_units, nonlinearity)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
        features = features.cuda()
        if sparse:
            sp_adj = sp_adj.cuda()
        else:
            adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0

    start1 = time.time()

    for epoch in range(nb_epochs):
        model.train()
        optimiser.zero_grad()

        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[:, idx, :]

        lbl_1 = torch.ones(batch_size, nb_nodes)
        lbl_2 = torch.zeros(batch_size, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)

        if torch.cuda.is_available():
            shuf_fts = shuf_fts.cuda()
            lbl = lbl.cuda()

        logits = model(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None)

        loss = b_xent(logits, lbl)

        print('Loss:', loss)

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'best_dgi.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early stopping!')
            break

        loss.backward()
        optimiser.step()

    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('best_dgi.pkl'))

    embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]

    train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    val_lbls = torch.argmax(labels[0, idx_val], dim=1)
    test_lbls = torch.argmax(labels[0, idx_test], dim=1)

    tot = torch.zeros(1)
    if torch.cuda.is_available():
        tot = tot.cuda()

    accs = []

    for _ in range(50):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        if torch.cuda.is_available():
            log.cuda()

        pat_steps = 0
        best_acc = torch.zeros(1)
        if torch.cuda.is_available():
            best_acc = best_acc.cuda()
        for _ in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)
        print(acc)
        tot += acc

    print('Average accuracy:', tot / 50)

    accs = torch.stack(accs)
    print(accs.mean())
    print(accs.std())

    DGI_time = time.time() - start1
    DGI_time_lst.append(DGI_time)
    print("DGI_time", DGI_time_lst)

    # print(features.shape, embeds.shape)
    # embeds = features
    embeds = embeds.cpu().numpy()
    embeds = embeds.reshape((embeds.shape[1], -1))

    org_features = features.cpu().numpy()
    org_features = org_features.reshape((org_features.shape[1], -1))

    print(embeds.shape)
    print(org_features.shape)

    # embeds = org_features

    color = L.astype(int)

    # start1 = time.time()
    # print("TSNE_euc start")
    # embeds_TSNE_euc = TSNE(n_components=2).fit(embeds)
    # elapsed_time1 = time.time() - start1
    # print(elapsed_time1)
    #
    # print("TSNE_cos start")
    # start2 = time.time()
    # embeds_TSNE_cos = TSNE(n_components=2, metric='cosine').fit(embeds)
    # elapsed_time2= time.time() - start2
    # print(elapsed_time2)
    #
    # print("UMAP_euc start")
    # start3 = time.time()
    # embeds_UMAP_euc = umap.UMAP(n_components=2, random_state=seed).fit_transform(embeds)
    # elapsed_time3 = time.time() - start3
    # print(elapsed_time3)

    print("UMAP_org start")
    start5 = time.time()
    embeds_UMAP = umap.UMAP(n_components=2, metric="cosine", random_state=seed).fit_transform(org_features)
    elapsed_time5 = time.time() - start5
    print(elapsed_time5)

    print('UMAP_cos start')
    start4 = time.time()
    embeds_UMAP_cos = umap.UMAP(n_components=2, metric="cosine", random_state=seed).fit_transform(embeds)
    elapsed_time4 = time.time() - start4
    UMAP_time_lst.append(elapsed_time4)
    print("UMAP_time", UMAP_time_lst)

    # embeds_UMAP_DGI = umap.UMAP(n_components=2, random_state=seed).fit_transform(embeds)
    emb_UMAP_lst.append(embeds_UMAP)
    emb_DGIUMAP_lst.append(embeds_UMAP_cos)

    file_path1 = 'embed_' + dataset + '_' + "UMAP_cos" + str(iter_number+1) + 'th'
    file_path2 = 'embed_' + dataset + '_' + "DGI-UMAP_cos" + str(iter_number+1) + 'th'
    np.savez(file_path1, emb=emb_UMAP_lst)
    np.savez(file_path2, emb=emb_DGIUMAP_lst)

# DGI = True
# model = 'DGI-UMAP' if DGI else "UMAP"
#
# file_path1 = 'embed_' + dataset + '_' + "UMAP_" + str(iter_n)
# file_path2 = 'embed_' + dataset + '_' + "DGI-UMAP_" + str(iter_n)
#
# A = adj.todense()
np.savez('cora_data', X = org_features, L = L, A=A)
# np.savez('cora', X = org_features, L = L, A=A)

# sns.set(context="paper", style="white")
#
# fig, ax = plt.subplots(figsize=(12, 10))
# color = L.astype(int)
# plt.scatter(
#     embeds_TSNE_DGI[:, 0], embeds_TSNE_DGI[:, 1], c=color, cmap="Spectral", s=10
# )
# plt.setp(ax, xticks=[], yticks=[])
# # plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)
#
# plt.show()

# fig, ax = plt.subplots(figsize=(12, 10))
# color = L.astype(int)
# plt.scatter(
#     embeds_UMAP[:, 0], embeds_UMAP[:, 1], c=color, cmap="Spectral", s=10
# )
# plt.setp(ax, xticks=[], yticks=[])
# # plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)
#
# plt.show()

def plot_embedding2D(node_pos, node_colors=None, di_graph=None):
    node_num, embedding_dimension = node_pos.shape
    if(embedding_dimension > 2):
        print("Embedding dimension greater than 2, use tSNE to reduce it to 2")
        model = umap.UMAP(n_components=2)
        node_pos = model.fit_transform(node_pos)

    if di_graph is None:
        # plot using plt scatter
        plt.scatter(node_pos[:, 0], node_pos[:, 1], c=node_colors)
    else:
        # plot using networkx with edge structure
        pos = {}
        for i in range(node_num):
            pos[i] = node_pos[i, :]
        if node_colors.any():
            nx.draw_networkx(di_graph, pos,
                                   node_color=node_colors,
                                   width=0.05, node_size=5,
                                   arrows=False, alpha=0.8,
                                   cmap='Spectral', with_labels=False)
        else:
            nx.draw_networkx(di_graph, pos, node_color=node_colors,
                             width=0.1, node_size=1, arrows=False,
                             alpha=0.8, with_labels=False, cmap='Spectral')

# # 隣接行列からグラフを構成
# G = nx.from_numpy_matrix(A)
#
# # 特徴行列Xがnode_pos
# node_pos = embeds_UMAP_DGI

# plot_embedding2D(node_pos, di_graph=G, node_colors=color)
# # plot_embedding2D(embeds_UMAP, di_graph=G, node_colors=color)
#
# fig, ax = plt.subplots(figsize=(12, 10))
# color = L.astype(int)
# plt.scatter(
#     embeds_UMAP_DGI[:, 0], embeds_UMAP_DGI[:, 1], c=color, cmap="Spectral", s=10
# )
# plt.setp(ax, xticks=[], yticks=[])
# # plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)
#
# plt.show()

# high-dimensional data: X = org_features, label: L, adj-Matrix_sparse: A_sparse = adj

# print("Result - Tsne_euc:")
# result_TSNE_euc = evaluate_viz_metrics(y_emb=embeds_TSNE_euc, X=org_features, L=L, A_sparse=adj, verbose=1)
# kmeans_acc_ari_ami_f1(X=embeds_TSNE_euc, L=L)
# # nearest_neighbours_generalisation_accuracy(embeds_UMAP_euc, L)
# # mantel_test(X=org_features, L=L, embed=embeds_TSNE_euc)
#
# print("Result - Tsne_cos:")
# result_TSNE_cos = evaluate_viz_metrics(y_emb=embeds_TSNE_cos, X=org_features, L=L, A_sparse=adj, verbose=1)
# kmeans_acc_ari_ami_f1(X=embeds_TSNE_cos, L=L)
# # nearest_neighbours_generalisation_accuracy(embeds_UMAP_cos, L)
# print("Result - UMAP_euc:")
# result_UMAP_euc = evaluate_viz_metrics(y_emb=embeds_UMAP_euc, X=org_features, L=L, A_sparse=adj, verbose=1)
# kmeans_acc_ari_ami_f1(X=embeds_UMAP_euc, L=L)
# # nearest_neighbours_generalisation_accuracy(embeds_UMAP_euc, L)
# # mantel_test(X=org_features, L=L, embed=embeds_UMAP_euc)
print("Result - UMAP_org:")
# result_UMAP_cos = evaluate_viz_metrics(y_emb=embeds_UMAP, X=org_features, L=L, A_sparse=adj, verbose=1)
kmeans_acc_ari_ami_f1(X=embeds_UMAP, L=L)
nearest_neighbours_generalisation_accuracy(embeds_UMAP, L)

print("Result - UMAP_cos:")
# result_UMAP_cos = evaluate_viz_metrics(y_emb=embeds_UMAP_cos, X=org_features, L=L, A_sparse=adj, verbose=1)
kmeans_acc_ari_ami_f1(X=embeds_UMAP_cos, L=L)
nearest_neighbours_generalisation_accuracy(embeds_UMAP_cos, L)

# score = nearest_neighbours_generalisation_accuracy(embeds_TSNE_euc, L)

# X_train, X_test, Y_train, Y_test = train_test_split(embeds_TSNE_euc, L, random_state=0)
# knc = KNeighborsClassifier(n_neighbors=1)
# knc.fit(X_train, Y_train)
# Y_pred = knc.predict(X_test)
# score = knc.score(X_test, Y_test)
# result_org.append(score)
# print("euclidean TSNE: 1-nn accuracy is ", result)

# X_train, X_test, Y_train, Y_test = train_test_split(embeds_TSNE_cos, L, random_state=0)
# knc = KNeighborsClassifier(n_neighbors=1)
# knc.fit(X_train, Y_train)
# Y_pred = knc.predict(X_test)
# score = knc.score(X_test, Y_test)
# result_org.append(score)
# print("cosine TSNE: 1-nn accuracy is ", score)

# X_train, X_test, Y_train, Y_test = train_test_split(embeds_UMAP_euc, L, random_state=0)
# knc = KNeighborsClassifier(n_neighbors=1)
# knc.fit(X_train, Y_train)
# Y_pred = knc.predict(X_test)
# score = knc.score(X_test, Y_test)
# result_org.append(score)
# print("euclidean UMAP: 1-nn accuracy is ", score)

# X_train, X_test, Y_train, Y_test = train_test_split(embeds_UMAP_cos, L, random_state=0)
# knc = KNeighborsClassifier(n_neighbors=1)
# knc.fit(X_train, Y_train)
# Y_pred = knc.predict(X_test)
# score = knc.score(X_test, Y_test)
# result_org.append(score)
# print("cosine UMAP: 1-nn accuracy is ", score)

# X_train, X_test, Y_train, Y_test = train_test_split(embeds_UMAP_DGI, L, random_state=0)
# knc = KNeighborsClassifier(n_neighbors=1)
# knc.fit(X_train, Y_train)
# Y_pred = knc.predict(X_test)
# score = knc.score(X_test, Y_test)
# # result_org.append(score)
# print("DGI UMAP: 1-nn accuracy is ", score)


# [80, 65, 51, 38, 92]

# Cora: [12, 75, 40, 39, 79, 48, 18, 84, 13, 28]
#   DGI_time [860.2302391529083, 645.2154319286346, 572.7620885372162, 535.3275208473206, 541.7825725078583, 668.4298441410065, 434.452317237854, 431.62584018707275, 873.8042180538177, 754.8982374668121]
#   UMAP_time [6.235907554626465, 6.24099063873291, 6.225750923156738, 6.228150844573975, 6.205959796905518, 6.200972557067871, 6.226802110671997, 6.212355613708496, 6.286243438720703, 6.203361749649048]
#   Epoch 276, 202, 177, 164, 167, 212, 130, 129, 293, 239,

# Citeseer: [35, 2, 29, 24, 76, 62, 89, 77, 93, 41]
#   DGI_time [550.5718469619751, 532.322455406189, 667.2490808963776, 735.2627582550049, 532.3537061214447, 793.1263172626495, 719.6570527553558, 597.8403031826019, 532.5540728569031, 734.6840817928314]
#   UMAP_time [7.691641569137573, 7.739134311676025, 7.714033603668213, 7.7038514614105225, 7.771690607070923, 7.732395172119141, 7.690489053726196, 7.710141181945801, 7.705132961273193, 7.706963062286377]
#   Epoch 118, 115, 150, 167, 115, 182, 163, 132, 115, 167,


# Pubmed: [3, 68, 71, 73, 22, 80, 81, 85, 59, 20]
#   DGI_time [2257.5849113464355, 1542.2274134159088, 3201.0049092769623, 2351.347795724869, 2225.073187828064, 2475.5281884670258, 2815.9435040950775, 2922.068704843521, 3106.4636600017548, 1609.6991441249847]
#   UMAP_time [30.569189071655273, 30.47438097000122, 29.93881893157959, 30.356384754180908, 30.67464303970337, 30.190309762954712, 29.846304655075073, 30.06774640083313, 30.57857394218445, 30.465651750564575]
#   Epoch 396, 264, 569, 413, 380, 414, 472, 490, 523, 266,