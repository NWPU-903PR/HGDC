# Preprocessing of Gene Expression

> In our study, we follow the data preprocessing steps as described in EMOGI[1], and the data preprocessing code is derived from [EMOGI](https://github.com/schulter/EMOGI). 
> 
> [1]Schulte-Sasse R, Budach S, Hnisz D, et al. Integration of multiomics data with graph convolutional networks to identify new 
cancer genes and their associated molecular mechanisms [J]. Nature Machine Intelligence, 2021, 3(6): 513-26.

The script `preprocess_gene_expression.py` preprocesses gene expression data from [Nature Scientific Data](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5903355/pdf/sdata201861.pdf).The gene expression data is already 
quantified (FPKM) and contains normal and tumor tissue samples from TCGA as well as normal tissue 
from GTEX. We only use the GTEX and TCGA tumor data to compute log2 fold changes of the two. The necessary data to run script `preprocess_gene_expression.py` can be downloaded from the website `https://figshare.com/articles/dataset/Data_record_3/5330593`. Taking the gene expression data of BLCA cancer as an example, users can run the `preprocess_gene_expression.py` script to generate a BLCA gene differential expression features (saved as `GE_BLCA_expression_matrix.tsv`)
