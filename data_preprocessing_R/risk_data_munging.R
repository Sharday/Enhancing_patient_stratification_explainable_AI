
#################################
# data munging for RISK data
#################################

################
# load library
################
library(sqldf)
library(gridExtra)
library(ggplot2)


##############################################
# set working directory to current directory
##############################################
setwd(dir = getwd())

##########################
# data munging
##########################


# data downloaded from GEO and data munging performed using COMMANDS_to_generate_ALLRISK.txt
# Contains ALL RISK
str_filename_RISK_RNASeq_withpath = "GSE57945_all_samples_RPKM_ALL_MOD.txt"

file_str_filename_RISK_RNASeq_withpath = read.csv(str_filename_RISK_RNASeq_withpath,
                                                  sep = '\t', header = TRUE,
                                                  stringsAsFactors=FALSE, na.strings="..", 
                                                  strip.white = TRUE)

#str_filename_RISK_RNASeq_withpath = "GSE57945_full_data_matrix_FULL_MOD.csv" # from Dominik and Nathan West, all patients (modified from original to remove top columns metadata)
# from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM1598408
# Illumina HiSeq 2000 for sequencing.
#Reads were aligned using TopHat, using hg19 as the reference genome and mapping reads per kilobase per million mapped reads (RPKM) as output.
#RPKM were normalized using the DESeq algorithm using Avaids NGS software, where normalized counts were log2-transformed and base-lined to the median expression of control samples.
#Normalized expression values were baselined to the median of all samples.
#We removed all transcripts that did not have at least 5 RPKM in at least 5 different samples.
#Genome_build: GRCh37 (hg19)

#file_str_filename_RISK_RNASeq_withpath = read.csv(str_filename_RISK_RNASeq_withpath,
#                                                sep = ',', header = TRUE,
#                                                stringsAsFactors=FALSE, na.strings="..", 
#                                                strip.white = TRUE)


# rename column names to be SQL compatible
names(file_str_filename_RISK_RNASeq_withpath)[1] <- "gene_id"
names(file_str_filename_RISK_RNASeq_withpath)[2] <- "gene_name"

# remove redundant column
file_str_filename_RISK_RNASeq_withpath$X <- NULL

# head(file_str_filename_RISK_RNASeq_withpath)

# all_genes_risk_scseq_array = sqldf(" select * 
#         from file_str_filename_RISK_RNASeq_withpath 
#         inner join res_df_same_direction_logfoldchange_cutoff 
#         on file_str_filename_RISK_RNASeq_withpath.gene_name = res_df_same_direction_logfoldchange_cutoff.gene_name
#     ")

#############################
# Upper Quartile normalize
#############################
# https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0206312#:~:text=Upper%20Quartile%20(UQ)%3A%20Under,multiplied%20by%20the%20mean%20upper


# source("~/periphery_project/bioinformatics/steve_notebooks/R_functions_stephen_refactored/upper_quartile_norm.R")
source('upper_quartile_norm.R')

# seelct all except first 2 columns which are text
file_str_filename_RISK_RNASeq_withpath_NORM = upperQuartileNormalise(as.matrix(file_str_filename_RISK_RNASeq_withpath[,c(-1,-2)]))

# log(TPM + 1)
file_str_filename_RISK_RNASeq_withpath_NORM = log2(file_str_filename_RISK_RNASeq_withpath_NORM + 1)

df_file_str_filename_RISK_RNASeq_withpath_NORM = as.data.frame(file_str_filename_RISK_RNASeq_withpath_NORM, 
                                                               stringAsFactors = FALSE)

#df_file_str_filename_RISK_RNASeq_withpath_NORM$gene_id <- file_str_filename_RISK_RNASeq_withpath$gene_id
df_file_str_filename_RISK_RNASeq_withpath_NORM$gene_name <- file_str_filename_RISK_RNASeq_withpath$gene_name


###################
# Perform PCA
###################
# source("~/periphery_project/bioinformatics/steve_notebooks/R_functions_stephen_refactored/pca.R")
source('pca.R')

temp_rawdata = as.matrix(file_str_filename_RISK_RNASeq_withpath[,c(-1,-2)])

summary(as.vector(temp_rawdata))
dim(temp_rawdata)

################################################################
# remove rows where genes are not expressed above a threshold
#   NOTE: threshold is arbitray
################################################################
i_max_threshold_filter_row = 10
idx2 <- apply(temp_rawdata, 1, max) > i_max_threshold_filter_row
temp_rawdata <- temp_rawdata[idx2, ]

# PCA analysis on raw data and transformations (if required: look at data first)
pca = prcomp(temp_rawdata)

#ggplot_prcomp(pca)

# get rotation
d <- as.data.frame(pca$rotation)

# PCA plot (not pretty)
plot(d$PC1,d$PC2)

# generate scree plot
head(pca$sdev)

(summary(pca))$importance["Proportion of Variance",]

plot( (summary(pca))$importance["Proportion of Variance",], 
      xlab="component",
      ylab="proportion of variance",
      main="scree plot")

# add a column for sample name OR stimulation condition etc (NOTE: d is a dataframe)
d$sample = rownames(d)

#gp <-  ggplot(d$sample, aes(d$PC1, d$PC2)) + geom_point(size=5)
#print(gp)

############################################################
# based on PCA analysis remove one outlier (get which one)
############################################################
idx_to_remove <- which(d$PC2 > 0.55)[1]

# remove that column
df_file_str_filename_RISK_RNASeq_withpath_NORM_PCAREMOVE = df_file_str_filename_RISK_RNASeq_withpath_NORM[,c(-idx_to_remove)]

# add gene names
# df_file_str_filename_RISK_RNASeq_withpath_NORM_PCAREMOVE$gene_id = df_file_str_filename_RISK_RNASeq_withpath_NORM$gene_id
df_file_str_filename_RISK_RNASeq_withpath_NORM_PCAREMOVE$gene_name = df_file_str_filename_RISK_RNASeq_withpath_NORM$gene_name

###########################
# save data frame to disk
###########################
write.csv(df_file_str_filename_RISK_RNASeq_withpath_NORM_PCAREMOVE,
          file = 'risk_cleaned.csv',
          row.names = FALSE,
          quote = FALSE
          )

# view the final data frame 
View(df_file_str_filename_RISK_RNASeq_withpath_NORM_PCAREMOVE)
View(file_str_filename_RISK_RNASeq_withpath)


