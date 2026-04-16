# A Landscape of Pharmacogenomic Interactions in Cancer
### Reproducing the ML part of the paper https://pubmed.ncbi.nlm.nih.gov/27397505/

Train many elastic net regressors on a variety of training data configurations, starting with mutation and gene expression data (all available publicly from https://cellmodelpassports.sanger.ac.uk/downloads), predicting IC50 (drug sensitivity).