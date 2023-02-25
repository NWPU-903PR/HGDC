import pandas as pd
import numpy as np
import os, sys

GTEX_NORMAL = True  # Use GTEx normal tissue gene expression or TCGA normal gene expression.
special_cancer_type = True
get_patient_from_barcode = lambda x: '-'.join(str(x).split('-')[:3])  # TCGA barcode for patient


# function to get the fold changes
def compute_geneexpression_foldchange(tumor_path, normal_path):
    """Preprocessing Gene Expression from DataDescriptor.

    Loading data for tissues and computing log2 fold changes
    The data can be loaded from the gzipped files directly. The fold changes can be computed as:
    \begin{equation*}
    FC_c = \log_2\big(\frac{median(P_c)}{median(N_t)}\big)
    \end{equation*}
    where $P_c \in \mathbb{R}^{n \times m}$ is the patient-wise matrix containing the FPKM-normalized read counts for
    $n$ patients (columns) and $m$ genes (rows) for a cancer type $c$. $N_t$ is the gtex normal tissue matrix for a
    tissue $t$ that corresponds to the cancer type $c$.

    For the normal data, we first compute the mean over all samples and then divide that by the FPKM counts for each
    patient in $P_c$. Only after that, we compute the mean across the patients.

    Parameters:
    ----------
    tumor_path:            Path to Tumor Sample Data
    normal_path:           Path to Normal Sample Data
    """
    # read tumor and normal data
    tumor_ge = pd.read_csv(tumor_path, compression='gzip', sep='\t').set_index('Hugo_Symbol').drop('Entrez_Gene_Id', axis=1)
    normal_ge = pd.read_csv(normal_path, compression='gzip', sep='\t').set_index('Hugo_Symbol').drop('Entrez_Gene_Id', axis=1)
    assert (np.all(tumor_ge.index == normal_ge.index))
    print(tumor_ge.shape, normal_ge.shape)

    # compute mean expression for tumor and normal. Then, compute log
    fc = tumor_ge.divide(normal_ge.median(axis=1), axis=0)
    print(fc.shape)

    log_fc = np.log2(fc)
    log_fc = log_fc.replace([np.inf, -np.inf], np.nan).dropna(axis=0)  # remove NaN and inf (from division by 0 or 0+eta)
    print("Dropped {} genes because they contained NaNs".format(fc.shape[0] - log_fc.shape[0]))
    return log_fc, tumor_ge, normal_ge


# get the file names ready
dir_name = ''
tumor_path = os.path.join(dir_name, '{}-rsem-fpkm-tcga-t.txt.gz')
gtex_path = os.path.join(dir_name, '{}-rsem-fpkm-gtex.txt.gz')

# now, apply that function for all TCGA-project and tissue pairs
if special_cancer_type:
    tissue_pairs = [('bladder', 'blca')]
else:
    tissue_pairs = [('bladder', 'blca'), ('breast', 'brca'), ('cervix', 'cesc'),
                    ('uterus', 'ucec'), ('colon', 'read'),
                    ('colon', 'coad'), ('liver', 'lihc'), ('salivary', 'hnsc'),
                    ('esophagus_mus', 'esca'), ('prostate', 'prad'), ('stomach', 'stad'),
                    ('thyroid', 'thca'), ('lung', 'luad'), ('lung', 'lusc'),
                    ('kidney', 'kirc'), ('kidney', 'kirp')]


log_fold_changes = []
tumor_fpkm = []
gtex_fpkm = []
tcga_normal_fpkm = []
mean_fold_changes = []
for gtex_tissue, tcga_project in tissue_pairs:
    fc_gtex, tumor, gtex = compute_geneexpression_foldchange(tumor_path=tumor_path.format(tcga_project),
                                                             normal_path=gtex_path.format(gtex_tissue)
                                                            )
    print(fc_gtex.shape)
    log_fold_changes.append(fc_gtex)
    mean_fold_changes.append(fc_gtex.median(axis=1))

mean_fold_changes = pd.DataFrame(mean_fold_changes, index=[i[1].upper() for i in tissue_pairs]).T
mean_fold_changes.fillna(0, inplace=True)

#Unify the order of cancer types in the column
if special_cancer_type:
    mean_fold_changes.columns = ['GE: ' + tissue_pairs[0][1].upper()]
    mean_fold_changes.to_csv('' + 'GE_' + tissue_pairs[0][1].upper() + '_expression_matrix.tsv', sep='\t')
else:
    CANCER_TYPES = ['KIRC', 'BRCA', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']
    mean_fold_changes = mean_fold_changes[CANCER_TYPES]
    CANCER_TYPES_GE = ['GE: KIRC', 'GE: BRCA', 'GE: READ', 'GE: PRAD', 'GE: STAD', 'GE: HNSC', 'GE: LUAD', 'GE: THCA', 'GE: BLCA',
                    'GE: ESCA', 'GE: LIHC', 'GE: UCEC', 'GE: COAD', 'GE: LUSC', 'GE: CESC', 'GE: KIRP']
    mean_fold_changes.columns = CANCER_TYPES_GE
    mean_fold_changes.to_csv('expression_mean_counts_16_gtexnormal.tsv', sep='\t')

