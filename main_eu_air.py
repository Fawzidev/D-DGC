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
from model_europe_air import FR_DGC
from datasets import format_data,sparse_mx_to_torch_sparse_tensor
from model_europe_air import clustering_metrics
from sklearn.mixture import GaussianMixture
from preprocessing import load_data, mask_test_edges, sparse_to_tuple, preprocess_graph, load_wiki, preprocess_graph2, laplacian, normalize, load_data_networks
from sklearn.metrics import confusion_matrix


# Dataset Name
dataset = "europe"
print("Europe dataset")
nClusters = 4
adj, features , labels = load_data_networks('europe', '/path/D_DGC/data/Airports/')
num_nodes = features.shape[1]
num_features = features.shape[0]

print(num_nodes)
print(num_features)

# Network parameters
num_neurons = 32
embedding_size = 16
save_path = "/path/D_DGC/results/"

# Pretraining parameters
epochs_pretrain = 200
lr_pretrain = 0.01

# Clustering parameters
epochs_cluster = 200
lr_cluster = 0.00001

beta1 = 0.01
beta2 = 0.005


#os.environ['CUDA_VISIBLE_DEVICES'] = ""
 
# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train

# Some preprocessing
adj_norm = preprocess_graph(adj)
num_nodes = adj.shape[0]
features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create Model
pos_weight_orig = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
adj_label_orig = adj_train + sp.eye(adj_train.shape[0])
adj_label_orig = sparse_to_tuple(adj_label_orig)
adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), torch.FloatTensor(adj_norm[1]), torch.Size(adj_norm[2]))
adj_label_orig = torch.sparse.FloatTensor(torch.LongTensor(adj_label_orig[0].T), torch.FloatTensor(adj_label_orig[1]), torch.Size(adj_label_orig[2]))
features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), torch.FloatTensor(features[1]), torch.Size(features[2]))
weight_mask_orig = adj_label_orig.to_dense().view(-1) == 1
weight_tensor_orig = torch.ones(weight_mask_orig.size(0))
weight_tensor_orig[weight_mask_orig] = pos_weight_orig



acc_array = []
network = FR_DGC(adj = adj_norm , num_neurons=num_neurons, num_features=num_features, embedding_size=embedding_size, nClusters=nClusters, activation="ReLU")
acc_array, y_pred, y = network.train(acc_array, adj_norm, features, adj_label_orig, labels, weight_tensor_orig, norm, optimizer="Adam", epochs=epochs_cluster, lr=lr_cluster, beta1=beta1, beta2=beta2, save_path=save_path, dataset=dataset)