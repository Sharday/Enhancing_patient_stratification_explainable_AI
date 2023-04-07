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


num_c = 4
x_train_scaled = pd.read_csv('../260_sample_train_scaled.csv').set_index("Patient_ID")
# x_train_scaled


x_test_scaled = pd.read_csv('../260_sample_test_scaled.csv').set_index("Patient_ID")
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
    n_inputs = 219
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
    encoder.load_weights("../encoder_ckpt")
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






def save_gmm(gmm, reduction_type):
    # save to file
    gmm_name = 'gmm_' + reduction_type
    np.save(gmm_name + '_weights', gmm.weights_, allow_pickle=False)
#     print('Saved ' + gmm_name + '_weights.npy')
    np.save(gmm_name + '_means', gmm.means_, allow_pickle=False)
    np.save(gmm_name + '_covariances', gmm.covariances_, allow_pickle=False)
    
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

# Autoencoder

def gmm_prediction_ae(x_test_scaled):
    # preprocessing

    full_dataset = full_ds.copy()
    split_pt = len(full_dataset) - len(x_test_scaled)
    full_dataset.iloc[split_pt:,:] = x_test_scaled

        
    # dim redction - autoencoder
    full_ae_dataset = ae_encode_dataset(full_dataset)
    
    # tsne
    tsne = manifold.TSNE(
        n_components=2,
        init="random",
        random_state=0,
        perplexity=24,
        n_iter=750,
        method='exact'
    )
    X = tsne.fit_transform(full_ae_dataset)
    X_test = X[split_pt:]
    
    
    filename_assignments = "autoencoder_assignments"
    load_gmm_name = "gmm_autoencoder"
    # load gmm model
    gmm = load_gmm(load_gmm_name)
    # load assignments
    with open(filename_assignments, "rb") as fp:   # Unpickling
        assignments = pickle.load(fp)
    return get_proba(gmm, assignments, X_test)

def gmm_model_get_prediction_ae(x_test_scaled):
    if len(x_test_scaled) <= len(full_ds):
        return gmm_prediction_ae(x_test_scaled)
    else:
        full_resp = np.empty((0,3))
        for i in range(0, len(x_test_scaled), len(full_ds)):
            end = i + len(full_ds)
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
    
    # dim reduction - PCA
    full_pca_dataset, _ = encode_pca(full_dataset)
    
    # tsne
    tsne = manifold.TSNE(
        n_components=2,
        init="random",
        random_state=0,
        perplexity=24,
        n_iter=750,
        method='exact'
    )
    X = tsne.fit_transform(full_pca_dataset)
    X_test = X[split_pt:]

    
    filename_assignments = "PCA_assignments"
    load_gmm_name = "gmm_PCA"
    # load gmm model
    gmm = load_gmm(load_gmm_name)
    # load assignments
    with open(filename_assignments, "rb") as fp:   # Unpickling
        assignments = pickle.load(fp)
    return get_proba(gmm, assignments, X_test)

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

