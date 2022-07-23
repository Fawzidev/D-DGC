#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authors : Nairouz Mrabah (mrabah.nairouz@courrier.uqam.ca) & Mohamed Fawzi Touati (touati.mohamed_fawzi@courrier.uqam.ca)
# @Link    : github.com/nairouz/GMM_VGAE
# @Paper   : Collaborative Graph Convolutional Networks: Unsupervised Learning Meets Semi-Supervised Learning
# @License : MIT License


import os
import torch
import metrics as mt
import numpy as np
import networkx as nx
import torch.nn as nn
import scipy.sparse as sp
import seaborn as sns
import torch.nn.functional as F
import itertools

from torch.autograd import Variable, grad
from tqdm import tqdm
from torch.optim import Adam, SGD, RMSprop
from torch.autograd import Variable
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import plot_confusion_matrix
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix
import os
import seaborn as sns

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from time import time

from sklearn import metrics
from munkres import Munkres, print_matrix
from preprocessing import sparse_to_tuple
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors

def random_uniform_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)

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

def q_mat(X, centers, alpha=1.0):
    if X.size == 0:
        q = np.array([])
    else:
        q = 1.0 / (1.0 + (np.sum(np.square(np.expand_dims(X, 1) - centers), axis=2) / alpha))
        q = q ** ((alpha + 1.0) / 2.0)
        q = np.transpose(np.transpose(q) / np.sum(q, axis=1))
    return q

def generate_unconflicted_data_index(emb, centers_emb, beta1, beta2):
    unconf_indices = []
    conf_indices = []
        
    q = q_mat(emb, centers_emb, alpha=1.0)
        
    confidence1 = q.max(1)
    #print("confidence1", confidence1)
    confidence2 = np.zeros((q.shape[0],))
    #print("confidence2", confidence2)
    
    #print("qshape", q.shape[0])
    a = np.argsort(q, axis=1)[:,-2]
    #print("a", a)
    for i in range(q.shape[0]):
        confidence2[i] = q[i,a[i]]
        if (confidence1[i]) > beta1 and (confidence1[i] - confidence2[i]) > beta2:
            unconf_indices.append(i)
        else:
            conf_indices.append(i)
    #print("confidence2", confidence2)
    unconf_indices = np.asarray(unconf_indices, dtype=int)
    conf_indices = np.asarray(conf_indices, dtype=int)
    return unconf_indices, conf_indices

def evaluate_links(adj, labels):
    nb_links = 0
    nb_false_links = 0
    nb_true_links = 0
    k = 0

    for line in adj:
        for j in range(line.indices.size):
            nb_links += 1 
            if labels[k] == labels[line.indices[j]]:
                nb_true_links += 1
            else:
                nb_false_links += 1
        k += 1
    assert nb_links == nb_true_links + nb_false_links
    print(" nb_links: " + str(nb_links) + " nb_true_links: " + str(nb_true_links) + "  nb_false_links: " + str(nb_false_links))  
    return nb_links, nb_false_links , nb_true_links

class clustering_metrics():
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label

    def clusteringAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]
            

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluationClusterModelFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()

        print('ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore))

        fh = open('recoder.txt', 'a')

        fh.write('ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore) )
        fh.write('\r\n')
        fh.flush()
        fh.close()

        return acc, nmi, adjscore, f1_macro, precision_macro, f1_micro, precision_micro

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation = F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = random_uniform_init(input_dim, output_dim) 
        self.activation = activation
        
    def forward(self, inputs, adj):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        outputs = self.activation(x)
        return outputs

class FR_DGC(nn.Module):

    def __init__(self, **kwargs):
        super(FR_DGC, self).__init__()
        self.num_neurons = kwargs['num_neurons']
        self.num_features = kwargs['num_features']
        self.embedding_size = kwargs['embedding_size']
        self.nClusters = kwargs['nClusters']
        if kwargs['activation'] == "ReLU":
            self.activation = F.relu
        if kwargs['activation'] == "Sigmoid":
            self.activation = F.sigmoid
        if kwargs['activation'] == "Tanh":
            self.activation = F.tanh

        # VGAE training parameters
        self.base_gcn = GraphConvSparse( self.num_features, self.num_neurons, self.activation)
        self.gcn_mean = GraphConvSparse( self.num_neurons, self.embedding_size, activation = lambda x:x)
        self.gcn_logstddev = GraphConvSparse( self.num_neurons, self.embedding_size, activation = lambda x:x)
       
        # GMM training parameters    
        self.pi = nn.Parameter(torch.ones(self.nClusters)/self.nClusters, requires_grad=True)
        self.mu_c = nn.Parameter(torch.randn(self.nClusters, self.embedding_size), requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.randn(self.nClusters, self.embedding_size),requires_grad=True)

    def pretrain(self, adj, features, adj_label, y, weight_tensor, norm, optimizer="Adam", epochs=200, lr=0.01, save_path="/content/drive/My Drive/Colab/PyTorch/GMM-VGAE/GMM_VGAE_BASE/Basic/results/", dataset="Cora"):
        if optimizer == "Adam":
            opti = Adam(self.parameters(), lr=lr)
        elif optimizer == "SGD":
            opti = SGD(self.parameters(), lr=lr, momentum=0.9)
        elif optimizer == "RMSProp":
            opti = RMSprop(self.parameters(), lr=lr)
        print('Pretraining......')
            
        # initialisation encoder weights
        epoch_bar = tqdm(range(epochs))
        acc_best = 0
        gmm = GaussianMixture(n_components = self.nClusters , covariance_type = 'diag')
        acc_list = []
        for _ in epoch_bar:
            opti.zero_grad()
            _,_, z = self.encode(features, adj)
            x_ = self.decode(z)
            loss = norm * F.binary_cross_entropy(x_.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
            loss.backward()
            opti.step()
            epoch_bar.write('Loss pretraining = {:.4f}'.format(loss))
            y_pred = gmm.fit_predict(z.detach().numpy())               
            self.pi.data = torch.from_numpy(gmm.weights_)
            self.mu_c.data = torch.from_numpy(gmm.means_)
            self.log_sigma2_c.data =  torch.log(torch.from_numpy(gmm.covariances_))
            acc = mt.acc(y, y_pred)
            acc_list.append(acc)
            if (acc > acc_best):
                acc_best = acc
                self.logstd = self.mean 
                torch.save(self.state_dict(), save_path + dataset + '/pretrain/model.pk')
        print("Best accuracy : ",acc_best)
        return acc_list

    def ELBO_Loss(self, features, adj, x_, adj_label, y, weight_tensor, norm, z_mu, z_sigma2_log, emb, L=1):
        pi = self.pi
        mu_c = self.mu_c
        log_sigma2_c = self.log_sigma2_c
        det = 1e-2 
        
        loss_recons = 1e-2 * features.size(0) * norm * F.binary_cross_entropy(x_.view(-1), adj_label, weight = weight_tensor) 
        
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(emb, mu_c,log_sigma2_c)) + det
        yita_c = yita_c / (yita_c.sum(1).view(-1,1))
        
        y_pred = self.predict1(emb)

        KL1 = 0.5 * torch.mean(torch.sum(yita_c * torch.sum(log_sigma2_c.unsqueeze(0)+
                                                torch.exp(z_sigma2_log.unsqueeze(1) - log_sigma2_c.unsqueeze(0)) +
                                                (z_mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2) / torch.exp(log_sigma2_c.unsqueeze(0)),2),1))
        KL2 = torch.mean(torch.sum(yita_c * torch.log(pi.unsqueeze(0) / (yita_c)), 1)) + 0.5 * torch.mean(torch.sum(1 + z_sigma2_log, 1))
        loss_clus = KL1 - KL2
        loss_elbo =  loss_recons + loss_clus 

        return loss_elbo, loss_recons, loss_clus 
   
    def train(self, acc_list, adj_norm, features, adj_label, y, weight_tensor, norm, optimizer="Adam", epochs=200, lr=0.01, beta1=0.30, beta2=0.15,  save_path="/content/drive/My Drive/Colab/PyTorch/GMM-VGAE/GMM_VGAE_BASE/Basic/results", dataset="Cora"):
        self.load_state_dict(torch.load(save_path + dataset + '/pretrain/model.pk'))
        if optimizer ==  "Adam":
            opti = Adam(self.parameters(), lr=lr, weight_decay = 0.089)
        elif optimizer == "SGD":
            opti = SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay = 0.01)
        elif optimizer == "RMSProp":
            opti = RMSprop(self.parameters(), lr=lr, weight_decay = 0.01)
        lr_s = StepLR(opti, step_size=10, gamma=0.9)
        
        import csv, os
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logfile = open(save_path + dataset + '/cluster/log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'f1_macro', 'f1_micro', 'precision_macro', 'precision_micro', 'acc_unconf', 'nmi_unconf', 'acc_conf', 'nmi_conf', 'nb_unconf', 'nb_conf', 'loss_recons', 'loss_clus' , 'loss', 'nb_l',  'nb_f_l', 'nb_t_l', 'nb_added_link_unconf_unconf', 'nb_false_added_link_unconf_unconf', 'nb_true_added_link_unconf_unconf', 'nb_deleted_link_unconf_unconf', 'nb_false_deleted_link_unconf_unconf', 'nb_true_deleted_link_unconf_unconf'])
        logwriter.writeheader()
        epoch_bar = tqdm(range(epochs))

        print('Training......')
        
        epoch_stable = 0 
        loss_list = []
        grad_loss_list = [] 
        previous_unconflicted = []
        previous_conflicted = []

        for epoch in epoch_bar:
            opti.zero_grad()
            z_mu, z_sigma2_log, emb = self.encode(features, adj_norm) 
            x_ = self.decode(emb)
            
            #with torch.no_grad():
            if epoch % 1 == 0 :
                unconflicted_ind, conflicted_ind = generate_unconflicted_data_index(emb.detach().numpy(), self.mu_c.detach().numpy(), beta1, beta2)
                #if epoch == 0:
                #    nb_l, nb_f_l, nb_t_l = evaluate_links(adj, y)
                #    adj, adj_label, weight_tensor, nb_added_link_unconf_unconf, nb_false_added_link_unconf_unconf, nb_true_added_link_unconf_unconf, nb_deleted_link_unconf_unconf, nb_false_deleted_link_unconf_unconf, nb_true_deleted_link_unconf_unconf =  self.update_graph(adj, y, emb, unconflicted_ind, conflicted_ind)

            if len(previous_unconflicted) < len(unconflicted_ind):
                z_mu = z_mu[unconflicted_ind]
                z_sigma2_log = z_sigma2_log[unconflicted_ind]
                emb_unconf = emb[unconflicted_ind]
                y_unconf = y[unconflicted_ind]
                acc_unconf, nmi_unconf, acc_conf, nmi_conf = self.compute_acc_and_nmi_conflicted_data(unconflicted_ind, conflicted_ind, emb, y)
                previous_conflicted = conflicted_ind
                previous_unconflicted = unconflicted_ind
            else :
                epoch_stable += 1
                z_mu = z_mu[previous_unconflicted]
                z_sigma2_log = z_sigma2_log[previous_unconflicted]
                emb_unconf = emb[previous_unconflicted]
                y_unconf = y[previous_unconflicted]
                acc_unconf, nmi_unconf, acc_conf, nmi_conf = self.compute_acc_and_nmi_conflicted_data(previous_unconflicted, previous_conflicted, emb, y)

            if epoch_stable >= 15:
                print("Update beta_1 and beta_2 \n") 
                epoch_stable = 0
                beta1 = beta1 * 0.95
                beta2 = beta2 * 0.85
                
            #if epoch % 50 == 0 and epoch <= 200:
            #    nb_l, nb_f_l, nb_t_l = evaluate_links(adj, y)
            #   adj, adj_label, weight_tensor, nb_added_link_unconf_unconf, nb_false_added_link_unconf_unconf, nb_true_added_link_unconf_unconf, nb_deleted_link_unconf_unconf, nb_false_deleted_link_unconf_unconf, nb_true_deleted_link_unconf_unconf =  self.update_graph(adj, y, emb, unconflicted_ind, conflicted_ind)
            nb_l, nb_f_l, nb_t_l = 0,0,0
            nb_added_link_unconf_unconf, nb_false_added_link_unconf_unconf, nb_true_added_link_unconf_unconf, nb_deleted_link_unconf_unconf, nb_false_deleted_link_unconf_unconf, nb_true_deleted_link_unconf_unconf =  0,0,0,0,0,0

            loss_elbo, loss_recons, loss_clus = self.ELBO_Loss(features, adj_norm, x_, adj_label.to_dense().view(-1), y_unconf, weight_tensor, norm, z_mu , z_sigma2_log, emb_unconf)
            epoch_bar.write('Loss={:.4f}'.format(loss_elbo.detach().numpy()))
            y_pred = self.predict1(emb)                            
            cm = clustering_metrics(y, y_pred)
            acc, nmi, adjscore, f1_macro, precision_macro, f1_micro, precision_micro = cm.evaluationClusterModelFromLabel()
            acc_list.append(acc)

            
            ##############################################################
            #Save logs 

            logdict = dict(iter=epoch, acc=acc, nmi=nmi, ari=adjscore, f1_macro=f1_macro , f1_micro=f1_micro, precision_macro=precision_macro, precision_micro=precision_micro,  acc_unconf=acc_unconf, nmi_unconf=nmi_unconf, acc_conf=acc_conf, nmi_conf=nmi_conf, nb_unconf=emb_unconf.size(0), nb_conf=emb.size(0) - emb_unconf.size(0), loss_recons = loss_recons.detach().numpy(), loss_clus = loss_clus.detach().numpy(), loss = loss_elbo.detach().numpy(), nb_l=nb_l,  nb_f_l=nb_f_l, nb_t_l=nb_t_l, nb_added_link_unconf_unconf=nb_added_link_unconf_unconf, nb_false_added_link_unconf_unconf=nb_false_added_link_unconf_unconf, nb_true_added_link_unconf_unconf=nb_true_added_link_unconf_unconf, nb_deleted_link_unconf_unconf=nb_deleted_link_unconf_unconf, nb_false_deleted_link_unconf_unconf=nb_false_deleted_link_unconf_unconf, nb_true_deleted_link_unconf_unconf=nb_true_deleted_link_unconf_unconf)
            logwriter.writerow(logdict)
            logfile.flush()
            
            ##############################################################
            # Update learnable parameters

            loss_elbo.backward()
            opti.step()
            lr_s.step()

            ##############################################################
            # Saving 2D Embedded space
            if epoch % 20 == 0:
                data = {'emb': emb.detach().numpy()}
                np.save(save_path + dataset + '/cluster/data_' + str(epoch) + '.npy', data)

            #if epoch % 20 == 0 :
            #    tsne = TSNE()
            #    tsne_results = tsne.fit_transform(emb.detach().numpy())
            #    plt.figure()
            #    sns.set(rc={'figure.figsize':(11.7,8.27)})
            #    palette = sns.color_palette("bright", self.nClusters)
            #    sns.set_style("white")
            #    clusterviz = sns.scatterplot(tsne_results[:,0], tsne_results[:,1], hue=y, legend='brief', palette=palette)
            #    plt.savefig(save_path + dataset + "/cluster/vis_tsne_" + str(epoch) + ".png", dpi=400)

        return acc_list, y_pred, y
               
    def gaussian_pdfs_log(self,x,mus,log_sigma2s):
        G=[]
        for c in range(self.nClusters):
            G.append(self.gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
        return torch.cat(G,1)

    def gaussian_pdf_log(self,x,mu,log_sigma2):
        c = -0.5 * torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1)
        return c

    def predict(self, features, adj):
        with torch.no_grad():
            _, _, z = self.encode(features, adj)
            pi = self.pi
            log_sigma2_c = self.log_sigma2_c  
            mu_c = self.mu_c
            det = 1e-2
            yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z,mu_c,log_sigma2_c)) + det
            yita = yita_c.detach().numpy()
            return np.argmax(yita, axis=1)

    def predict1(self, z):
        with torch.no_grad():
            pi = self.pi
            log_sigma2_c = self.log_sigma2_c  
            mu_c = self.mu_c
            det = 1e-2
            yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))+det
            yita = yita_c.detach().numpy()
            return np.argmax(yita, axis=1)

    def encode(self, x_features, adj):
        hidden = self.base_gcn(x_features, adj)
        self.mean = self.gcn_mean(hidden, adj)
        self.logstd = self.gcn_logstddev(hidden, adj)
        gaussian_noise = torch.randn(x_features.size(0), self.embedding_size)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return self.mean, self.logstd, sampled_z
            
    @staticmethod
    def decode(z):
        A_pred = torch.sigmoid(torch.matmul(z,z.t()))
        return A_pred
        
    def compute_acc_and_nmi_conflicted_data(self, unconf_indices, conf_indices, emb, y, save_path="/content/drive/My Drive/Colab/PyTorch/GMM-VGAE/results/", dataset="Cora"):
        if unconf_indices.size == 0:
            print(' '*8 + "Empty list of unconflicted data")
            acc_unconf = 0
            nmi_unconf = 0
        else:
            print("Number of Unconflicted points : ", len(unconf_indices))
            emb_unconf = emb[unconf_indices]
            y_unconf = y[unconf_indices]
            y_pred_unconf = self.predict1(emb_unconf)
            acc_unconf = mt.acc(y_unconf, y_pred_unconf)
            nmi_unconf = mt.nmi(y_unconf, y_pred_unconf)
            print(' '*8 + '|==>  acc unconflicted data: %.4f,  nmi unconflicted data: %.4f  <==|'% (acc_unconf, nmi_unconf))

        if conf_indices.size == 0:
            print(' '*8 + "Empty list of conflicted data")
            acc_conf = 0
            nmi_conf = 0
        else:
            print("Number of conflicted points : ", len(conf_indices))
            emb_conf = emb[conf_indices] 
            y_conf = y[conf_indices]
            y_pred_conf = self.predict1(emb_conf)
            acc_conf = mt.acc(y_conf, y_pred_conf)
            nmi_conf = mt.nmi(y_conf, y_pred_conf)
            print(' '*8 + '|==>  acc conflicted data: %.4f,  nmi conflicted data: %.4f  <==|'% (mt.acc(y_conf, y_pred_conf), mt.nmi(y_conf, y_pred_conf)))    
        return acc_unconf, nmi_unconf, acc_conf, nmi_conf

    def generate_centers(self, emb_unconf):
        # Plus proche voisin unconflictuel de centre correspondant aux differents points unconflictuels 
        y_pred = self.predict1(emb_unconf)
        nn = NearestNeighbors(n_neighbors= 1, algorithm='ball_tree').fit(emb_unconf.detach().numpy())
        _, indices = nn.kneighbors(self.mu_c.detach().numpy())
        return indices[y_pred]