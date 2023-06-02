# Enhancing patient stratification and interpretability through class-contrastive and feature attribution techniques: unveiling potential therapeutic gene targets

Main files created from scratch:
* 1_gene_sampling_heatmaps.ipynb - sampling from the RISK dataset
* 2_preprocessing_training_autoencoder.ipynb - training the autoencoder
* 3_kmeans_gmm_eval.ipynb - clustering and classification using GMMs and KMeans, including evaluation
* gene_shap_minimal/4_gene_shap_dependent.ipynb - producing SHAP explainability plots to identify and rank risk genes by phenotype, using SHAP adapted for feature dependence
* gene_shap_minimal/4_gene_shap_independent.ipynb - producing SHAP explainability plots to identify and rank risk genes by phenotype, using original SHAP with feature independence
* gene_shap_minimal/5_consensus_clustering_gene_modules_deep_ulcer.ipynb - identification and characterisation of gene modules for Crohn's disease with deep ulcer
* gene_shap_minimal/5_consensus_clustering_gene_modules_no_ulcer.ipynb - identification and characterisation of gene modules for Crohn's disease without deep ulcer
* gene_shap_minimal/6_class_contrastive_GMM_gene_modules.ipynb - class-contrastive technique for cluster explainability based on identified gene modules
* gene_shap_minimal/6_class_contrastive_GMM_volcano_plot.ipynb - class-contrastive technique for cluster explainability based on gene sets extracted from volcano plot

This code builds upon the SHAP functionality. The original SHAP implementation can be found at https://github.com/slundberg/shap. A minimal version is provided using the .py files in gene_shap_minimal. 
This file was modified to incorporate feature dependence into the method:
* gene_shap_minimal/explainers/_kernel.py
The following files were created from scratch for facilitation:
* gene_shap_minimal/kmeans_gmm_eval_fns.py - includes functions to get GMM predictions
* gene_shap_minimal/objects.py - functions to return required objects
* gene_shap_minimal/calc_shap.py - calculates and stores SHAP values
