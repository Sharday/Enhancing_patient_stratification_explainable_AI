# from fn_file import get_number, get_spec_number
from kmeans_gmm_eval_fns import gmm_model_get_prediction_ae, gmm_model_get_prediction_pca
import pandas as pd

# print(get_number())
# print(get_spec_number())

x_test_scaled = pd.read_csv('../260_sample_test_scaled.csv').set_index("Patient_ID")
gmm_ae_prob = gmm_model_get_prediction_ae(x_test_scaled)
gmm_pca_prob = gmm_model_get_prediction_pca(x_test_scaled)
print("AE predictions:")
print(gmm_ae_prob)
print("PCA predictions:")
print(gmm_pca_prob)