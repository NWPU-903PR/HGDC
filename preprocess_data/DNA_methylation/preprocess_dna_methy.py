import pandas as pd
import numpy as np
import os, sys

USE_BATCH_CORRECTED = True  # Whether or not to use the batch corrected data or not
USE_FOLDCHANGES = False  #Compute log fold changes between tumor and normal or simply subtract the normal methylation
                        # from the cancer case. The latter does not result in a significant change between 2% and 4%
                        # methylation (both basically None) while the fold changes would significantly change that.
special_cancer_type = True
def get_ratio(tumor_sample_path, normal_sample_path, use_foldchange=False):
    """Computing Differential DNA Methylation.

    This function can compute differential DNA methylation values (subtracted normal and tumor beta values) for
    promoters and gene bodies of genes.


    Parameters:
    ----------
    tumor_sample_path:                      Path to a cancer type tumor sample matrix
    normal_sample_path:                     Path to a cancer type normal sample matrix
    use_foldchange:                         Compute log fold changes between tumor and normal or simply subtract the
                                            normal methylation from the cancer case.
    """

    tumor_samples = pd.read_csv(tumor_sample_path, sep='\t')
    normal_samples = pd.read_csv(normal_sample_path, sep='\t')

    # restore the column names for samples
    tumor_samples.columns = ['-'.join(i.split('.')[:7]) + '|' + i.split('.')[7] + '|' + i.split('.')[8] for i in tumor_samples.columns]
    normal_samples.columns = ['-'.join(i.split('.')[:7]) + '|' + i.split('.')[7] + '|' + i.split('.')[8] for i in normal_samples.columns]

    # make it the same genes for both dataframes and verify
    common_genes = tumor_samples.index.intersection(normal_samples.index)
    tumor_samples = tumor_samples.reindex(common_genes)
    normal_samples = normal_samples.reindex(common_genes)
    assert ((normal_samples.index == tumor_samples.index).all())

    if use_foldchange:
        # compute log2 fold changes (for each sample, divide by mean normal values. Then compute mean across samples)
        fc = np.log2(tumor_samples.divide(normal_samples.mean(axis=1), axis=0))
        fc_nan = fc.replace([np.inf, -np.inf], np.nan)
        print ("Got {} invalid values after computing log2 fold changes".format(fc_nan.isnull().sum().sum()))
        ratio = fc_nan.dropna(axis=0) # remove NaN and inf (from division by 0 or 0+eta)
    else:
        # subtract the ratios from one another
        ratio = tumor_samples.subtract(normal_samples.mean(axis=1), axis=0)
        ratio_nan = ratio.replace([np.inf, -np.inf], np.nan)
        print ("Got {} invalid values after subtraction".format(ratio_nan.isnull().sum().sum()))
        ratio = ratio_nan.dropna(axis=0)

    return ratio

base_dir = '../data/pancancer/TCGA/DNA_methylation/pan_cancer/'

if USE_BATCH_CORRECTED:
    fname = '{}_samples.adjusted.tsv'
else:
    fname = '{}_samples.tsv'

all_ratios = []
for ctype in os.listdir(base_dir):
    ctype_dir = os.path.join(base_dir, ctype)
    if os.path.isdir(ctype_dir) and not ctype == 'OV': # don't have normals for OV
        ratio = get_ratio(os.path.join(ctype_dir, fname.format('tumor')),
                          os.path.join(ctype_dir, fname.format('normal')),
                          use_foldchange=USE_FOLDCHANGES
                         )
        all_ratios.append(ratio)
        print ("Processed {}".format(ctype))

meth_all_samples = all_ratios[0].join(all_ratios[1:]).dropna(axis=0)

# first, transpose and add column with cancer type
meth_t = meth_all_samples.T
meth_t['cancer_type'] = [i[1] for i in meth_t.index.str.split('|')]

# compute mean across all cancer types
meth_mean = meth_t.fillna(0).groupby('cancer_type').mean().T
meth_mean.columns = [i.upper() for i in meth_mean.columns]

if special_cancer_type:
    meth_mean.columns = ['METH: ' + ctype]
    fname = '../data/pancancer/TCGA/' + 'METH_' + ctype + '_methylation_{}_mean.tsv'.format('FC' if USE_FOLDCHANGES else 'RATIO')
    meth_mean.to_csv(fname, sep='\t')
    print("Mean methylation written to: {}".format(fname))
else:
    CANCER_TYPES = ['KIRC', 'BRCA', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']
    meth_mean = meth_mean[CANCER_TYPES]
    CANCER_TYPES_METH = ['METH: KIRC', 'METH: BRCA', 'METH: READ', 'METH: PRAD', 'METH: STAD', 'METH: HNSC', 'METH: LUAD', 'METH: THCA', 'METH: BLCA',
                    'METH: ESCA', 'METH: LIHC', 'METH: UCEC', 'METH: COAD', 'METH: LUSC', 'METH: CESC', 'METH: KIRP']
    meth_mean.columns = CANCER_TYPES_METH

    fname = '../data/pancancer/TCGA/methylation_{}_mean.tsv'.format('FC' if USE_FOLDCHANGES else 'RATIO')
    meth_mean.to_csv(fname, sep='\t')
    print ("Mean methylation written to: {}".format(fname))
