import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from models import DGI, LogReg
from utils import process

from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# dataset = 'cora'
dataset = 'citeseer'
# dataset = 'pubmed'

# training params
batch_size = 1
nb_epochs = 10000
patience = 20
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 256 if dataset == 'pubmed' else 512

sparse = True
nonlinearity = 'prelu' # special name to separate parameters

adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
features, _ = process.preprocess_features(features)

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


# print(features.shape, embeds.shape)
# embeds = features
embeds = embeds.cpu().numpy()
embeds = embeds.reshape((embeds.shape[1], -1))

org_features = features.cpu().numpy()
org_features = org_features.reshape((org_features.shape[1], -1))

print(embeds.shape)
print(org_features.shape)

embeds_TSNE_DGI = TSNE(n_components=2).fit_transform(embeds)
# embeds_UMAP = umap.UMAP(n_components=2).fit_transform(org_features)
embeds_UMAP_DGI = umap.UMAP(n_components=2).fit_transform(embeds)

sns.set(context="paper", style="white")

fig, ax = plt.subplots(figsize=(12, 10))
color = L.astype(int)
plt.scatter(
    embeds_TSNE_DGI[:, 0], embeds_TSNE_DGI[:, 1], c=color, cmap="Spectral", s=10
)
plt.setp(ax, xticks=[], yticks=[])
# plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)

plt.show()

# fig, ax = plt.subplots(figsize=(12, 10))
# color = L.astype(int)
# plt.scatter(
#     embeds_UMAP[:, 0], embeds_UMAP[:, 1], c=color, cmap="Spectral", s=10
# )
# plt.setp(ax, xticks=[], yticks=[])
# # plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)
#
# plt.show()

fig, ax = plt.subplots(figsize=(12, 10))
color = L.astype(int)
plt.scatter(
    embeds_UMAP_DGI[:, 0], embeds_UMAP_DGI[:, 1], c=color, cmap="Spectral", s=10
)
plt.setp(ax, xticks=[], yticks=[])
# plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)

plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(embeds_TSNE_DGI, L, random_state=0)
knc = KNeighborsClassifier(n_neighbors=1)
knc.fit(X_train, Y_train)
Y_pred = knc.predict(X_test)
score = knc.score(X_test, Y_test)
# result_org.append(score)
print("DGI TSNE: 1-nn accuracy is ", score)

# X_train, X_test, Y_train, Y_test = train_test_split(embeds_UMAP, L, random_state=0)
# knc = KNeighborsClassifier(n_neighbors=1)
# knc.fit(X_train, Y_train)
# Y_pred = knc.predict(X_test)
# score = knc.score(X_test, Y_test)
# # result_org.append(score)
# print("Original UMAP: 1-nn accuracy is ", score)

X_train, X_test, Y_train, Y_test = train_test_split(embeds_UMAP_DGI, L, random_state=0)
knc = KNeighborsClassifier(n_neighbors=1)
knc.fit(X_train, Y_train)
Y_pred = knc.predict(X_test)
score = knc.score(X_test, Y_test)
# result_org.append(score)
print("DGI UMAP: 1-nn accuracy is ", score)