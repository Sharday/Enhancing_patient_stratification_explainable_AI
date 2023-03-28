#!/bin/sh

# get file from GEO website
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE57nnn/GSE57945/suppl/GSE57945_all_samples_RPKM.txt.gz

# unzip file
gzip -d GSE57945_all_samples_RPKM.txt.gz

# remove characters
tr '\r' '\t' < GSE57945_all_samples_RPKM.txt > GSE57945_all_samples_RPKM_ALL_MOD.txt

# call R script to perform data munging
R --no-save < risk_data_munging.R


