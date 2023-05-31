import math
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredLogarithmicError
import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
import pickle
np.random.seed(0)
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv1D, Conv2D, LeakyReLU, MaxPool1D, AveragePooling1D, UpSampling1D, Flatten, Dense, Reshape, BatchNormalization
from tensorflow.keras import initializers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from scipy.stats import multivariate_normal
from keras.optimizers import Adam, SGD, Adadelta
from sklearn.metrics import accuracy_score, f1_score
import objects
import time
from testp import progressBar
from kmeans_gmm_eval_fns import gmm_model_get_prediction_ae
import sys


x_train_scaled = pd.read_csv('../data/260_sample_train_scaled.csv').set_index("Patient_ID")
x_test_scaled = pd.read_csv('../data/260_sample_test_scaled.csv').set_index("Patient_ID")




explainer = objects.get_explainer(model=gmm_model_get_prediction_ae, data=x_train_scaled, link="logit", 
                                  vis=False, feature_dependence=True)

shap_values = explainer.shap_values(X=x_test_scaled)
print("final shap values:",shap_values)

with open("../data/models/shap/all_dep_2p11_2p5_mean", "wb") as fp:   #Pickling
    pickle.dump(shap_values, fp)


