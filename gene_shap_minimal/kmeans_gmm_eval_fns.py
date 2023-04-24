#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
# import keras_tuner.tuners as kt
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

import numpy as np
import pickle
np.random.seed(0)
from tensorflow.keras.layers import Conv1D, Conv2D, LeakyReLU, MaxPool1D, AveragePooling1D, UpSampling1D, Flatten, Dense, Reshape, BatchNormalization
# https://towardsdatascience.com/improve-your-model-performance-with-auto-encoders-d4ee543b4154
from sklearn.decomposition import PCA


import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = '1'


num_c = 4
x_train_scaled = pd.read_csv('../data/260_sample_train_scaled.csv').set_index("Patient_ID")
# x_train_scaled


x_test_scaled = pd.read_csv('../data/260_sample_test_scaled.csv').set_index("Patient_ID")
# x_test_scaled



full_ds = pd.concat([x_train_scaled, x_test_scaled])
# full_ds



test_set = x_test_scaled.copy()



def encode_pca(dataset):
    pca_x_test = PCA(n_components=32)
    principalComponents_x_test = pca_x_test.fit_transform(dataset)
    if isinstance(dataset,np.ndarray):
        index = None
    else:
        index=dataset.index
    pca_x_test_ds = pd.DataFrame(data = principalComponents_x_test, 
                                       index=index)
    return pca_x_test_ds, pca_x_test

def load_encoder():
    n_inputs = 220
    n_bottleneck = 32
    encoder = Sequential(
                [

    #                 Input(shape=(n_inputs,)),
                    # encoder level 1
                    Dense(n_inputs*2),
                    BatchNormalization(),
                    LeakyReLU(),
                    # encoder level 2
                    Dense(n_inputs),
                    BatchNormalization(),
                    LeakyReLU(),
                    # bottleneck
                    Dense(n_bottleneck)
                ]
            )

    sh = test_set.head(1).shape
    encoder.load_weights("../data/models/cd_encoder")
    encoder.build(sh) 
#     encoder.summary()
    return encoder




    
def ae_encode_dataset(dataset):
    encoder = load_encoder()
    latent_var = np.arange(32)
    if isinstance(dataset,np.ndarray):
        recon = encoder(dataset)
        index=None
    else:
        recon = encoder(dataset.values)
        index = dataset.index
    r = pd.DataFrame(recon, columns=latent_var, index=index)
    return r
    


import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from time import time

from matplotlib.patches import Ellipse
from sklearn import mixture


import scipy.stats
from scipy.stats import multivariate_normal



def get_probs(point, comps, assignments, w):
    probs = []
    for elem in assignments:
        p_cand = []
        for cluster in elem:
            p = comps[cluster].pdf(point) * w[cluster]
#             print("p:",p)
            p_cand.append(p)
#         print(p_cand)
        p = max(p_cand)
        probs.append(p)
        
    return probs

def sum_to_one(vals): 
#     print(t, u)
    return vals / vals.sum()


# In[58]:


def get_proba(gmm, assignments, X_test):
    # GMM mixture component distributions
    mu, covar, w = gmm.means_, gmm.covariances_, gmm.weights_
    # print("means:",mu)
    # print("covariances_:",covar)
    # print("weights_:",w)
    comps = [multivariate_normal(mu[i], covar[i]) for i in range(num_c)]

    X_test = X_test.tolist()
    new_list = []
    for point in X_test:
        probs = get_probs(point, comps, assignments, gmm.weights_)
        new_list.append(probs)
    pdf_vals = np.asarray(new_list) * 100000 # TEMP

    # rescale so adds up to 1
    proba = pdf_vals.copy()
    # for i in range(len(proba)):
    #     row = proba[i,:]
    #     rescaled = sum_to_one(row)

    #     proba[i,:] = rescaled
    
    return proba


def pair_repeat(dup, cls_assignments, num_c, amounts):
    all_clusters = np.arange(num_c)
    # replace one of the duplicates with max of remaining clusters
    replace_class = np.where(cls_assignments == dup[0])[0][0]
    rem_cluster_a, rem_cluster_b = np.setdiff1d(all_clusters,cls_assignments)
    # rem_cluster_b
    if amounts[replace_class][rem_cluster_a] > amounts[replace_class][rem_cluster_b]:
        cls_assignments[replace_class] = rem_cluster_a
    else:
        cls_assignments[replace_class] = rem_cluster_b
        
    return cls_assignments

def handle_duplicates(cls_assignments, dup, c, num_c, amounts):
#     print("c:",c)
    if len(c) != 1:
        
        cls_assignments = pair_repeat(dup, cls_assignments, num_c, amounts)
    else: # 3 repeats of same thing
        # reassign class least associated to next most associated cluster
        given_cluster = dup[0]
        cls_least = np.argmin(amounts[:,given_cluster])
#         print(cls_least)
        class_amounts = amounts[cls_least,:]
        class_amounts[given_cluster] = -1
        next_cluster = np.argmax(class_amounts)
        cls_assignments[cls_least] = next_cluster
#         print("intermediate assignments",cls_assignments)
        
        # handle other duplicate pair
        u, c = np.unique(cls_assignments, return_counts=True)
        dup = u[c > 1]
        cls_assignments = pair_repeat(dup, cls_assignments, num_c, amounts)

    return cls_assignments




# def save_gmm(gmm, reduction_type):
#     # save to file
#     gmm_name = 'gmm_' + reduction_type
#     np.save(gmm_name + '_weights', gmm.weights_, allow_pickle=False)
# #     print('Saved ' + gmm_name + '_weights.npy')
#     np.save(gmm_name + '_means', gmm.means_, allow_pickle=False)
#     np.save(gmm_name + '_covariances', gmm.covariances_, allow_pickle=False)
    
def load_gmm(gmm_name):
    # reload
    means = np.load(gmm_name + '_means.npy')
    covar = np.load(gmm_name + '_covariances.npy')
    loaded_gmm = mixture.GaussianMixture(n_components = len(means), covariance_type='full')
    loaded_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covar))
    loaded_gmm.weights_ = np.load(gmm_name + '_weights.npy')
    loaded_gmm.means_ = means
    loaded_gmm.covariances_ = covar
    
    return loaded_gmm

# def softmax(x): 
#     return np.exp(x)/sum(np.exp(x))

def get_class_probs(X_test, gmm, assignments, couple):
    clus_probs = sum_to_one(gmm.predict_proba(X_test))
    num_classes = 3
    
    arr = np.zeros((X_test.shape[0], num_classes))
    for c in range(num_classes):
        if assignments[c] == couple:
            arr[:,c] = np.max(clus_probs[:,couple], axis=1)
        else:
            cluster = assignments[c][0]
            arr[:,c] = clus_probs[:,cluster]
    arr = sum_to_one(arr)
    return np.argmax(arr, axis=1), arr

# Autoencoder


# def gmm_prediction_ae(x_test_scaled):
#     # preprocessing

#     full_dataset = full_ds.copy()
#     split_pt = len(full_dataset) - len(x_test_scaled)
#     full_dataset.iloc[split_pt:,:] = x_test_scaled

        
#     # dim redction - autoencoder
#     full_ae_dataset = ae_encode_dataset(full_dataset)
    
#     # tsne
#     tsne = manifold.TSNE(
#         n_components=2,
#         init="random",
#         random_state=0,
#         perplexity=100,
#         n_iter=750,
#         method='exact'
#     )
#     X = tsne.fit_transform(full_ae_dataset)
#     X_test = X[split_pt:]
    
#     path = "../data/models/"
#     filename_assignments = path + "autoencoder_assignments"
#     filename_couple = path + "autoencoder_couple"
#     load_gmm_name = path + "gmm_autoencoder"
#     # load gmm model
#     gmm = load_gmm(load_gmm_name)
#     # load assignments
#     with open(filename_assignments, "rb") as fp:   # Unpickling
#         assignments = pickle.load(fp)
#     with open(filename_couple, "rb") as fp:   # Unpickling
#         couple = pickle.load(fp)
#     _, probs = get_class_probs(X_test, gmm, assignments, couple)
#     return probs

def fit_gmm(full_dataset, perplexity, num_c, split_pt=None):
    # print("full_dataset:",full_dataset)
    gmm = mixture.GaussianMixture(n_components=num_c,covariance_type='full', random_state=42)
    tsne = manifold.TSNE(
        n_components=2,
        init="random",
        random_state=0,
        perplexity=perplexity,
        n_iter=750,
        method='exact'
    )
    X = tsne.fit_transform(full_dataset)
    # print("split:",split_pt)

    
    if split_pt is None:
        split_pt = int(0.7*len(full_dataset))
        split_pt_val = int(0.85*len(full_dataset))
        X_test = X[split_pt_val:]
        X_val = X[split_pt:split_pt_val]
    else:
        X_val = None
        X_test = X[split_pt:]
    X_train = X[:split_pt]
    gmm.fit(X_train)
    
    return gmm, X_train, X_test, X_val


def get_amount_matrix_gmm(gmm, X_train, train_disease_labels):
    mu, covar, w = gmm.means_, gmm.covariances_, gmm.weights_
    # print("means:",mu)
    # print("covariances_:",covar)
    # print("weights_:",w)
    
    comps = [multivariate_normal(mu[i], covar[i]) for i in range(num_c)]
    
    # make matrix to record amount of each class in each mixture component
    amounts = np.zeros((3, num_c))
    
    
    classes = ["control", "CD_no_ulcer", "CD_deep_ulcer"]
    for c, cls in enumerate(classes):
        for i, comp in enumerate(comps): # add up contributions for each component
            weight = w[i]
            pts = X_train[train_disease_labels==c]
            curr_sum = np.sum(comp.pdf(pts)) * weight
            amounts[c][i] = curr_sum
    # print(amounts)
    #                cluster 0, cluster 1, cluster 2, cluster 3
    # control 
    # CD no ulcer
    # CD deep ulcer
    
    return amounts

def process_clusters(amounts, X_train, num_c):
    all_clusters = np.arange(num_c)

    
    cls_assignments = np.argmax(amounts, axis=1) # assigned to class 0, 1, 2
#     print("initial cls assignments:",cls_assignments)
    
    # check for and handle duplicates
    u, c = np.unique(cls_assignments, return_counts=True)
    dup = u[c > 1]
#     print("dup:",dup[0])
    if len(dup) > 0:
        cls_assignments = handle_duplicates(cls_assignments, dup, c, num_c, amounts)
        
            
    class_assignment_amounts = np.max(amounts, axis=1) 
    
    assignments = [None] * 3
    
    assigned = 0
    while assigned < num_c - 1:
        curr_max_class = np.argmax(class_assignment_amounts)
        assigned_cluster = cls_assignments[curr_max_class]
        if assignments[curr_max_class] is None:
            assignments[curr_max_class] = [assigned_cluster]
        else:
            assignments[curr_max_class].append(assigned_cluster)
        class_assignment_amounts[curr_max_class] = -1
        assigned += 1

    # Assign remaining cluster
    
    rem_cluster = np.setdiff1d(all_clusters,cls_assignments)[0]
    
    rem_cls_assignment = np.argmax(amounts[:,rem_cluster], axis=0)
    assignments[rem_cls_assignment].append(rem_cluster)
    couple = assignments[rem_cls_assignment]
    # clusters assigned to disease class 0, 1, 2 (control, CD_no_ulcer, CD_deep_ulcer)
    
#     print(assignments)
#     print(couple)
    return assignments, couple

def final_gmm_model_get_clusters(gmm, X_train, X_test, train_disease_labels):
    # process
    amounts = get_amount_matrix_gmm(gmm, X_train, train_disease_labels)
    assignments, couple = process_clusters(amounts, X_train, num_c)
    # retrieve clusters
#     gmm_labels = gmm.predict(X_test)
#     test_set_clusters = get_final_clusters(assignments, couple, gmm_labels)
    test_set_clusters, probs = get_class_probs(X_test, gmm, assignments, couple)
#     print(test_set_clusters)
#     print(probs)
    
    return test_set_clusters, probs

def gmm_prediction_ae(x_test_scaled):
    num_c = 4
    full_dataset = full_ds.copy()
    split_pt = len(full_dataset) - len(x_test_scaled)
    full_dataset.iloc[split_pt:,:] = x_test_scaled

    # format disease labels
    with open("../data/d_labels", "rb") as fp:   # Unpickling
        full_disease_labels = pickle.load(fp)
    train_disease_labels = full_disease_labels[:split_pt]
    test_disease_labels = full_disease_labels[split_pt:]
    
    split_pt = None if split_pt == 0 else split_pt
    full_ae_dataset = ae_encode_dataset(full_dataset)
    perplexity = 130
    gmm, X_train, X_test, X_val = fit_gmm(full_ae_dataset, perplexity, num_c, split_pt=split_pt) # tSNE -> 2D pts

    if X_val is None:
        X = np.concatenate([X_train, X_test])
    else:
        X = np.concatenate([X_train, X_val, X_test])
        
    X_test = X[split_pt:]

    test_set_clusters, probs = final_gmm_model_get_clusters(gmm, X_train, X_test,train_disease_labels)
    return probs


def gmm_model_get_prediction_ae(x_test_scaled):
    full_length = 260
    if len(x_test_scaled) <= full_length:
        return gmm_prediction_ae(x_test_scaled)
    else:
        full_resp = np.empty((0,3))
        for i in range(0, len(x_test_scaled), full_length):
            end = i + full_length
            if end > len(x_test_scaled):
                end = len(x_test_scaled)
            next_resp = gmm_prediction_ae(x_test_scaled[i:end])
            full_resp = np.concatenate([full_resp, next_resp])
        return full_resp

# PCA

def gmm_prediction_pca(x_test_scaled):
    # preprocessing
    full_dataset = full_ds.copy()
    split_pt = len(full_dataset) - len(x_test_scaled)
    full_dataset.iloc[split_pt:,:] = x_test_scaled
    
    # dim redction - PCA
    full_pca_dataset, _ = encode_pca(full_dataset)
    
    # tsne
    tsne = manifold.TSNE(
        n_components=2,
        init="random",
        random_state=0,
        perplexity=250,
        n_iter=750,
        method='exact'
    )
    X = tsne.fit_transform(full_pca_dataset)
    X_test = X[split_pt:]

    path = "../data/models/"
    filename_assignments = path + "PCA_assignments"
    filename_couple = path + "PCA_couple"
    load_gmm_name = path + "gmm_PCA"
    # load gmm model
    gmm = load_gmm(load_gmm_name)
    # load assignments
    with open(filename_assignments, "rb") as fp:   # Unpickling
        assignments = pickle.load(fp)
    with open(filename_couple, "rb") as fp:   # Unpickling
        couple = pickle.load(fp)
    _, probs = get_class_probs(X_test, gmm, assignments, couple)
    return probs

def gmm_model_get_prediction_pca(x_test_scaled):
    if len(x_test_scaled) <= len(full_ds):
        return gmm_prediction_pca(x_test_scaled)
    else:
        full_resp = np.empty((0,3))
        for i in range(0, len(x_test_scaled), len(full_ds)):
            end = i + len(full_ds)
            if end > len(x_test_scaled):
                end = len(x_test_scaled)
            next_resp = gmm_prediction_pca(x_test_scaled[i:end])
            full_resp = np.concatenate([full_resp, next_resp])
        return full_resp

# def gmm_model_get_prediction_ae(x_test_scaled):


#     # preprocessing
#     # load full dataset
#     smaller_set = False
#     if len(x_test_scaled) < len(full_ds):
#         smaller_set = True
#         full_dataset = full_ds.copy()
#         split_pt = len(full_dataset) - len(x_test_scaled)
#         full_dataset.iloc[split_pt:,:] = x_test_scaled
#     else:
#         full_dataset = x_test_scaled.copy()
        
#     # dim reduction - autoencoder
#     full_ae_dataset = ae_encode_dataset(full_dataset)
#     print("ae encoded")
    
#     # tsne
#     tsne = manifold.TSNE(
#         n_components=2,
#         init="random",
#         random_state=0,
#         perplexity=24,
#         n_iter=750,
#         method='exact'
#     )
#     X = tsne.fit_transform(full_ae_dataset)
#     if smaller_set:
#         X_test = X[split_pt:]
#     else:
#         X_test = X
#     print("done tsne")
    
    
#     filename_assignments = "autoencoder_assignments"
#     load_gmm_name = "gmm_autoencoder"
#     # load gmm model
#     gmm = load_gmm(load_gmm_name)
#     # load assignments
#     with open(filename_assignments, "rb") as fp:   # Unpickling
#         assignments = pickle.load(fp)
#     print("getting proba")
#     return get_proba(gmm, assignments, X_test)

# def gmm_model_get_prediction_pca(x_test_scaled):
#     smaller_set = False
#     if len(x_test_scaled) < len(full_ds):
#         smaller_set = True
#         full_dataset = full_ds.copy()
#         split_pt = len(full_dataset) - len(x_test_scaled)
#         full_dataset.iloc[split_pt:,:] = x_test_scaled
#     else:
#         full_dataset = x_test_scaled.copy()
    
#     # dim redction - PCA
#     full_pca_dataset, _ = encode_pca(full_dataset)
    
#     # tsne
#     tsne = manifold.TSNE(
#         n_components=2,
#         init="random",
#         random_state=0,
#         perplexity=24,
#         n_iter=750,
#         method='exact'
#     )
#     X = tsne.fit_transform(full_pca_dataset)
#     if smaller_set:
#         X_test = X[split_pt:]
#     else:
#         X_test = X
    
    
#     filename_assignments = "PCA_assignments"
#     load_gmm_name = "gmm_PCA"
#     # load gmm model
#     gmm = load_gmm(load_gmm_name)
#     # load assignments
#     with open(filename_assignments, "rb") as fp:   # Unpickling
#         assignments = pickle.load(fp)
#     return get_proba(gmm, assignments, X_test)



# def final_gmm_model_get_clusters(gmm, X_train, X_test, train_disease_labels):
#     # process
#     amounts = get_amount_matrix_gmm(gmm, X_train, train_disease_labels)
#     assignments, couple = process_clusters(amounts, X_train, num_c)
#     # retrieve clusters
# #     gmm_labels = gmm.predict(X_test)
# #     test_set_clusters = get_final_clusters(assignments, couple, gmm_labels)
#     test_set_clusters, probs = get_class_probs(X_test, gmm, assignments, couple)
# #     print(test_set_clusters)
# #     print(probs)
    
#     return test_set_clusters, probs