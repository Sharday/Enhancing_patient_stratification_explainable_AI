from explainers._kernel import Kernel as KernelExplainer
from _explanation import Explanation
# from _explanation import Explanation
from utils._clustering import hclust
import numpy as np

def get_explainer(model, data, link):
    print("getting explainer")

    # explainer = KernelExplainer(lambda x: np.zeros(x.shape[0]), np.ones((2, 4)), nsamples=100)
    explainer = KernelExplainer(model, data, link)
    # e = explainer.explain(np.ones((1, 4)))
    # assert np.sum(np.abs(e)) < 1e-8

    return explainer

def get_explanation(values, base_values, data, feature_names):
    print("getting explanation")
    return Explanation(values=values, base_values=base_values, data=data, feature_names=feature_names)

def get_clustering(ds, labels):
    return hclust(ds, labels)

# say_hello()