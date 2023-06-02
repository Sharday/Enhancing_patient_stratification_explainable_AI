from explainers._kernel import Kernel as KernelExplainer
from _explanation import Explanation
from utils._clustering import hclust
import numpy as np
import sys

def get_explainer(model, data, vis=False, link=None, num_instances=None, specific_indices=None, feature_dependence=True):

    explainer = KernelExplainer(model, data, vis=vis, link=link, num_instances=num_instances, feature_dependence=feature_dependence, specific_indices=specific_indices)

    return explainer

def get_explanation(values, base_values, data, feature_names):
    return Explanation(values=values, base_values=base_values, data=data, feature_names=feature_names)

def get_clustering(ds, labels):
    return hclust(ds, labels)
