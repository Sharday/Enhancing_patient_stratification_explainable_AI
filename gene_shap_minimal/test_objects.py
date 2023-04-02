import math
import pandas as pd
import tensorflow as tf
# import keras_tuner.tuners as kt
import matplotlib.pyplot as plt
import keras
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredLogarithmicError
import numpy as np
import pickle
np.random.seed(0)
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv1D, Conv2D, LeakyReLU, MaxPool1D, AveragePooling1D, UpSampling1D, Flatten, Dense, Reshape, BatchNormalization
# https://towardsdatascience.com/improve-your-model-performance-with-auto-encoders-d4ee543b4154
from tensorflow.keras import initializers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from scipy.stats import multivariate_normal
from keras.optimizers import Adam, SGD, Adadelta
import objects
import time
from testp import progressBar

x_train_scaled = pd.read_csv('../260_sample_train_scaled.csv').set_index("Patient_ID")
x_test_scaled = pd.read_csv('../260_sample_test_scaled.csv').set_index("Patient_ID")

patient_ids_train = np.array(x_train_scaled.index)
patient_ids_test = np.array(x_test_scaled.index)

def classify(x):
    if "_control" in x: # control
        return 0
    elif "CD_plain" in x: # Crohn's Disease no deep ulcer
#         print(x)
        return 1
    elif "CD_deep_ulcer" in x: # Crohn's Disease deep ulcer
#         print(x)
        return 2
    else:
        return 3 # Ulcerative Collitis

vec = np.vectorize(classify)

disease_labels_train = vec(patient_ids_train)
disease_labels_test = vec(patient_ids_test)



# # A List of Items
# items = list(range(0, 100))

# # A Nicer, Single-Call Usage
# for item in progressBar(items, prefix = 'Progress:', suffix = 'Complete', length = 50):
#     # Do stuff...
#     time.sleep(0.1)

# clustering = objects.get_clustering(x_test_scaled, disease_labels_test)
compound_model = keras.models.load_model('cd_clf')


# explainer = objects.get_explainer(model=compound_model.predict, data=x_train_scaled, link="logit", specific_indices=[41])
explainer = objects.get_explainer(model=compound_model.predict, data=x_train_scaled, link="logit")
# explainer = objects.get_explainer(model=f_dummy, data=x_train_scaled.iloc[:15])

shap_values = explainer.shap_values(X=x_test_scaled)
print("final shap values:",shap_values)

with open("shap_values_builtin_tl_219", "wb") as fp:   #Pickling
    pickle.dump(shap_values, fp)