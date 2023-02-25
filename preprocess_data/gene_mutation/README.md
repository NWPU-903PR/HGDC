# Preprocessing Mutation Frequencies
> In our study, we follow the data preprocessing steps as described in EMOGI[1], and the data preprocessing code is derived from [EMOGI](https://github.com/schulter/EMOGI). 
> 
> [1]Schulte-Sasse R, Budach S, Hnisz D, et al. Integration of multiomics data with graph convolutional networks to identify new 
cancer genes and their associated molecular mechanisms [J]. Nature Machine Intelligence, 2021, 3(6): 513-26.

## Preprocessing of SNVs
The script `preprocess_mutation_freqs.py` preprocesses single nucleotide variants from MAF files.
The script can normalize SNV frequencies for exonic gene length if GENCODE annotation is provided.
The final result return a mean mutation frequencies per cancer type (gene x cancer type matrix).
We provide an example in the script `preprocess_mutation_freqs.py` to generate a BLCA single-nucleotide mutation features(saved as `MF_BLCA_mutation_matrix.tsv`). Users should download the relevant MAF files from TCGA before running `preprocess_mutation_freqs.py`. Taking BLCA as an example, the MAF file `TCGA.BLCA.mutect.0e239d8f-47b0-4e47-9716-e9ecc87605b9.DR-10.0.somatic.maf.gz` is necessary to generate `MF_BLCA_mutation_matrix.tsv`. The names of all MAF files used in our study are listed in the file `gdc_manifest.mutations.mutect2.2018-11-23.txt`. Before running `preprocess_mutation_freqs.py`, users should extract file `gencode.v28.basic.annotation.gff3` from the GZIP file `gencode.v28.basic.annotation.gff3.gz`.

