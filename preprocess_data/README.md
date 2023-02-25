# Preprocessing of TCGA data
> In our study, we follow the data preprocessing steps as described in EMOGI[1], and the data preprocessing code is derived from [EMOGI](https://github.com/schulter/EMOGI). 
> 
> [1]Schulte-Sasse R, Budach S, Hnisz D, et al. Integration of multiomics data with graph convolutional networks to identify new 
cancer genes and their associated molecular mechanisms [J]. Nature Machine Intelligence, 2021, 3(6): 513-26.

This folder contains scripts to process and normalize:

* [Mutation MAF files](mutfreq/README.md) (can be downloaded via TCGA data portal),
* [DNA methylation data](methylation/README.md) from Illumina 450k bead arrays (can be downloaded again from TCGA data portal)
* [Gene expression data](expression/README.md) (normalized data available in the publication from "[Data Descriptor: Unifying cancer and normal RNA sequencing data from different sources](https://figshare.com/articles/dataset/Data_record_3/5330593)" Wang et al., 2018).


When running scripts, users should ensure that the storage path of relevant data files is consistent with the settings in the code.
