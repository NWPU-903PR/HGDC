# Preprocessing DNA Methylation Data

> In our study, we follow the data preprocessing steps as described in EMOGI[1], and the data preprocessing code is derived from [EMOGI](https://github.com/schulter/EMOGI). 
> 
> [1]Schulte-Sasse R, Budach S, Hnisz D, et al. Integration of multiomics data with graph convolutional networks to identify new 
cancer genes and their associated molecular mechanisms [J]. Nature Machine Intelligence, 2021, 3(6): 513-26.

The scripts and code in this directory let you preprocess DNA methylation data from 450k Illumina bead arrays.
The script `preprocess_dna_methy.py` can show you how to combine tumor and normal DNA methylation data
while the script `get_mean_sample_meth.py` lets you compute the average promoter and gene body methylation
given some genome annotation.To generate your own differential DNA methylation feature matrix, use the script and code 
provided in this file to run the sequence as follows.
 
- First, use the gdc-client tool to download data from the two provided gdc 
files (normal samples and tumor samples). The instructions are as follows：
	`gdc-client download -m gdc_manifest.2018-11-16.txt`
- Then, use the script `get_mean_sample_meth.py` to calculate the respective methylation matrices of tumor samples and normal samples, 
- Next, use the notebook `batchcorrection.ipynb` and an R script to calculate the two matrices respectively Perform the batch effect operation, 
- Finally, use the script `preprocess_dna_methy.py` to obtain the differential methylation matrix.

## Compute Mean Promoter Methylation
There are multiple options to compute the promoter DNA methylation and the script allows for multiple ones. We decided to
define a promoter based on the most 5\`transcript for that gene. We then define the promoter as the region +-1000 base pairs
around the TSS. The script works by first computing a map where each measured CpG site is assigned to a gene together with the
distance to the TSS. We then use that mapping and apply it to all samples to reduce runtime.
You can run the script using:

```
python get_mean_sample_meth.py --annotation <path-to-annotation-gff3 file> --methylation-dir <path-to-downloaded-TCGA-methylation-data> --output <path-to-gene-sample-matrix>
```

## Batch Correction
We provided a notebook `batchcorrection.ipynb` and an R script to do batch correction using
[ComBat](https://bioconductor.org/packages/release/bioc/vignettes/sva/inst/doc/sva.pdf). Noting that, users should make sure all dependency packages for the R script are installed before doing the batch correction. We use the plate numbers as batch variables to normalize against and normalize each study individually.

## Computing Differential DNA Methylation
The script `preprocess_dna_methy.py` allows you to compute differential DNA methylation values (subtracted
normal and tumor beta values) for promoters and gene bodies of genes.

We provide an example to generate a BLCA differential DNA methylation features(METH_BLCA_methylation_RATIO_mean.tsv). However, due to the large size of relevant files of methylation expression data, user should download relevant files from TCGA as described above. The files of `gdc_manifest.dna_meth.solidnormal.2018-11-16.txt` and `gdc_manifest.dna_meth.tumor.2018-11-16.txt` contains names of all files used in our study.
