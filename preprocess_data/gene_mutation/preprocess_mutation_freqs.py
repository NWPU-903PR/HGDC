import pandas as pd
import numpy as np
import os, sys, math
import numpy.matlib

# Before running `preprocess_mutation_freqs.py`,
# users should extract file `gencode.v28.basic.annotation.gff3` from
# the GZIP file `gencode.v28.basic.annotation.gff3.gz`.

# This value tells us how many mutations a sample is allowed to have before
# we remove it. The reason behind this filtering step is that there are some
# samples with accumulated mutations that do not neccessarily contribute
ULTRA_MUTATED_SAMPLES_THRESHOLD = 1000
NORMALIZE_FOR_GENE_LENGTH = True  # Whether dividing SNV frequencies by the length of the gene or not

special_cancer_type = True # special_cancer or pan_cancer

gencode_annotation_path = 'gencode.v28.basic.annotation.gff3'  # Path to GENCODE annotation file
snv_base_dir = './mut_data/' # Single nucleotide mutation data path downloaded from TCGA
ultra_mutated_samples_path = 'ultramutated_tumor_ids.txt' # Ultramutated tumor sample ID file path
trim_fun = lambda x: '-'.join(str(x).split('-')[:4])  # TCGA barcode including sample


def get_gene_sample_matrix(path, ultra_mutated_samples_path, gene_lengths=None):
    """Preprocessing SNVs from TCGA.

    This function processed Mutation Annotation Format(MAF) files from TCGA, following the preprocessing pipeline from HotNet2.
    The execution steps are as follows:

    1.Loading the MAF as DataFrame
    2.Removing non-silent mutations
    3.Removing ultra-mutators
    4.Compute Gene×Sample matrix Sc for each cancer type c
    5.Normalize for the gene length using the GENCODE basic annotation

    Parameters:
    ----------
    path:                                  Path to a cancer type MAF file downloaded from TCGA
    ultra_mutated_samples_path:            Ultramutated tumor sample ID file path
    gene_lengths:                          Normalize for gene length if gene_lengths is not None
    """

    maf = pd.read_csv(path, compression='gzip', sep='\t', comment='#', header=0)
    non_silent = maf[maf.Variant_Classification != 'Silent']  # remove silent mutations
    # non_silent = maf
    print("Removed {} (of {}) mutations because they are silent".format(maf.shape[0] - non_silent.shape[0],
                                                                        maf.shape[0]))
    non_silent.Tumor_Sample_Barcode = non_silent.Tumor_Sample_Barcode.map(trim_fun)

    # remove ultra-mutators
    ultra_mutated_samples = pd.read_csv(ultra_mutated_samples_path, header=None, names=['Tumor_IDs'])
    ultra_mutated_samples.Tumor_IDs = ultra_mutated_samples.Tumor_IDs.map(trim_fun)
    maf_no_ultra = non_silent[~non_silent.Tumor_Sample_Barcode.isin(ultra_mutated_samples.Tumor_IDs)]
    print("Left with {} SNVs after removing {} mutations in hyper-mutated samples".format(maf_no_ultra.shape[0],
                                                                                          non_silent.shape[0] -
                                                                                          maf_no_ultra.shape[0]))

    # compute gene x sample matrix
    gene_barcode_mat = maf_no_ultra.groupby(['Hugo_Symbol', 'Tumor_Sample_Barcode']).size().reset_index().rename(
        columns={0: 'count'})
    assert ((gene_barcode_mat.pivot(index='Hugo_Symbol',
                                    columns='Tumor_Sample_Barcode',
                                    values='count'
                                    ).sum() == maf_no_ultra.groupby('Tumor_Sample_Barcode').count().Hugo_Symbol).all())
    assert ((gene_barcode_mat.pivot(index='Hugo_Symbol',
                                    columns='Tumor_Sample_Barcode',
                                    values='count'
                                    ).sum(axis=1) == maf_no_ultra.groupby(
        'Hugo_Symbol').count().Tumor_Sample_Barcode).all())
    gene_sample_matrix = gene_barcode_mat.pivot(index='Hugo_Symbol', columns='Tumor_Sample_Barcode',
                                                values='count').replace(np.NaN, 0)

    # lastly, normalize for gene length if requested
    if not gene_lengths is None:
        bpmr = gene_sample_matrix.sum() / float(gene_lengths.sum())  # mutation rate per base per patient (vector)
        expected_mutation_rates = pd.DataFrame(
            np.matlib.repmat(bpmr, gene_lengths.shape[0], 1) * gene_lengths.values.reshape(-1, 1),
            index=gene_lengths.index,
            columns=bpmr.index
            )  # expected mutation frequency per gene and patient
        # normalized mutation frequency (actual mutations divided by expected mutations)
        # We add 1 to the dividend to avoid division by 0 or too much influence of short genes
        denom = 1 + expected_mutation_rates.reindex_like(gene_sample_matrix).fillna(
            expected_mutation_rates.median(axis=0))
        normalized_gene_sample_matrix = gene_sample_matrix / denom
        return normalized_gene_sample_matrix
    else:
        return gene_sample_matrix

annotation_gencode = pd.read_csv(gencode_annotation_path,
                                 comment='#', sep='\t', skiprows=7,header=None,
                                 names=['chr', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attr']
                                )
# derive length of all exons
annotation_gencode = annotation_gencode[annotation_gencode.type == 'exon']
annotation_gencode['length'] = np.abs(annotation_gencode.end - annotation_gencode.start)

# extract the gene name for each exon
def get_hugo_symbol(row):
    s = row.attr
    for elem in s.split(';'):
        if elem.startswith('gene_name'):
            return elem.split('=')[1].strip()
    return None

annotation_gencode['Hugo_Symbol'] = annotation_gencode.apply(get_hugo_symbol, axis=1)
# add length of exons together for each gene
exonic_gene_lengths = annotation_gencode.groupby('Hugo_Symbol').length.sum()

all_matrices = []
ctypes = []
for dname in os.listdir(snv_base_dir):
    ctype_dir = os.path.join(snv_base_dir, dname)
    if os.path.isdir(ctype_dir):
        for maffile in os.listdir(ctype_dir):
            if maffile.endswith('.maf.gz'):
                gene_sample_matrix = get_gene_sample_matrix(os.path.join(ctype_dir, maffile),
                                                            ultra_mutated_samples_path=ultra_mutated_samples_path,
                                                            gene_lengths=exonic_gene_lengths if NORMALIZE_FOR_GENE_LENGTH else None
                                                           )
                all_matrices.append(gene_sample_matrix)
                ctype = maffile.split('.')[1]
                print ("Processed {} with {} samples".format(ctype, gene_sample_matrix.shape[1]))
                ctypes.append(ctype)

# get the mean matrices

mean_mutations = []
for df in all_matrices:
    mean_mutations.append(df.mean(axis=1))
mean_mutation_matrix = pd.DataFrame(mean_mutations, index=ctypes).T
mean_mutation_matrix.fillna(0, inplace=True)
if special_cancer_type:
    mean_mutation_matrix.columns = ['MF: ' + ctype]
    mean_mutation_matrix.to_csv('' + 'MF_' + ctypes[0] + '_mutation_matrix.tsv', sep='\t')
else:
    #Unify the order of cancer types in the column
    CANCER_TYPES = ['KIRC', 'BRCA', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']
    mean_mutation_matrix = mean_mutation_matrix[CANCER_TYPES]
    CANCER_TYPES_MF = ['MF: KIRC', 'MF: BRCA', 'MF: READ', 'MF: PRAD', 'MF: STAD', 'MF: HNSC', 'MF: LUAD', 'MF: THCA', 'MF: BLCA',
                    'MF: ESCA', 'MF: LIHC', 'MF: UCEC', 'MF: COAD', 'MF: LUSC', 'MF: CESC', 'MF: KIRP']
    mean_mutation_matrix.columns = CANCER_TYPES_MF
    mean_mutation_matrix.to_csv('mean_mutation_matrix.tsv', sep='\t')