import sys
import os
import numpy as np
import torch
import torch.nn as nn
import metrics
import scipy.sparse as sp
import matplotlib.pyplot as plt
import torch.nn.functional as F
import itertools
import math
from time import time
from model_cora import FR_DGC
from datasets import format_data,sparse_mx_to_torch_sparse_tensor
from model_cora import clustering_metrics
from sklearn.mixture import GaussianMixture
from preprocessing import load_data, mask_test_edges, sparse_to_tuple, preprocess_graph, load_wiki, preprocess_graph2, laplacian, normalize
from sklearn.metrics import confusion_matrix

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

def map_vector_to_clusters(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_true[i], y_pred[i]] += 1
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    y_true_mapped = np.zeros(y_pred.shape)
    for i in range(y_pred.shape[0]):
        y_true_mapped[i] = col_ind[y_true[i]]
    return y_true_mapped.astype(int)

def plot_confusion_matrix(cm, target_names, title='Matrice de confusion', cmap=None, normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.1f}".format(round_half_up(cm[i, j], decimals=1)),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('Label de référence')
    plt.xlabel('Label prédit \n Précision={:0.4f}; Mal classés = {:0.4f}'.format(accuracy, misclass))
    #plt.show()
    plt.savefig("/path/D_DGC/results/Cora/cluster/confusion_matrix.png", dpi=400)

# Dataset Name
dataset = "Cora"
print("Cora dataset")
feas = format_data('cora', '/path/D_DGC/data/Cora')
num_nodes = feas['features'].size(0)
num_features = feas['features'].size(1)
nClusters = 7
adj, features , labels = load_data('cora', '/path/D_DGC/data/Cora')

# Network parameters
num_neurons = 32
embedding_size = 16
save_path = "/path/D_DGC/results/"

# Pretraining parameters
epochs_pretrain = 200
lr_pretrain = 0.01

# Clustering parameters
epochs_cluster = 200
lr_cluster = 0.01
beta1 = 0.3
beta2 = 0.15

#os.environ['CUDA_VISIBLE_DEVICES'] = ""
 
adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
adj.eliminate_zeros()
#adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
#adj = adj_train
adj_norm = preprocess_graph(adj)
features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
pos_weight_orig = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
adj_label = adj + sp.eye(adj.shape[0])
adj_label = sparse_to_tuple(adj_label)
adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), torch.FloatTensor(adj_norm[1]), torch.Size(adj_norm[2]))
adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), torch.FloatTensor(adj_label[1]), torch.Size(adj_label[2]))
features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), torch.FloatTensor(features[1]), torch.Size(features[2]))
weight_mask_orig = adj_label.to_dense().view(-1) == 1
weight_tensor_orig = torch.ones(weight_mask_orig.size(0))
weight_tensor_orig[weight_mask_orig] = pos_weight_orig

plt.figure()
acc_array = []

#acc_array = network.pretrain(adj_norm, features, adj_label_orig, labels, weight_tensor_orig, norm, optimizer="Adam", epochs=epochs_pretrain, lr=lr_pretrain, save_path=save_path, dataset=dataset)
network = FR_DGC(adj=adj_norm, num_neurons=num_neurons, num_features=num_features, embedding_size=embedding_size, nClusters=nClusters, activation="ReLU")
acc_array, y_pred, y = network.train(acc_array, adj_norm, features, adj_label, labels, weight_tensor_orig, norm, optimizer="Adam", epochs=epochs_cluster, lr=lr_cluster, beta1=beta1, beta2=beta2, save_path=save_path, dataset=dataset)

target_names = ["0", "1", "2", "3", "4", "5", "6"]
y_mapped = map_vector_to_clusters(y, y_pred)
cm = confusion_matrix(y_true=y_mapped, y_pred=y_pred, normalize='true')
plot_confusion_matrix(cm, target_names, title='Matrice de confusion', normalize=True, cmap=plt.cm.Blues)