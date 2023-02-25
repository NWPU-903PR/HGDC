import pandas as pd
import numpy as np
import torch
import pickle
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

def load_network(file_path):
    """
    Load network from file.
    :param file_path: Full pathname of the network file
    :return: net (class: pandas.DataFrame): Edges in the network, nodes (class: pandas.DataFrame): The nodes in the network
    """
    net = pd.read_table(filepath_or_buffer=file_path, header=None,
                        index_col=None, names=['source', 'target'], sep='\t')
    nodes = pd.concat([net['source'], net['target']], ignore_index=True)
    nodes = pd.DataFrame(nodes, columns=['nodes']).drop_duplicates()
    nodes.reset_index(drop=True, inplace=True)
    return net, nodes

def build_customized_feature_matrix(feat_file_lst, network_file, feat_name_lst):
    """
    Build feature matrix on your own data.
    :param feat_file_lst: List of full pathnames of feature files. Each feat_file in feat_file_lst contains two columns, i.e., gene names and feature values.
    :param network_file: Full pathname of network file
    :param feat_name_lst: List of feature names
    :return: Concatenated feature matrix with n rows(genes) and m columns(features) (class: pandas.DataFrame)
    """
    feat_dic = dict()
    # Load gene features from each feat_file
    for i in range(0, len(feat_file_lst)):
        feat_dic[feat_name_lst[i]] = pd.read_csv(feat_file_lst[i], sep='\t', index_col=0)
    # Load network from file
    net, net_nodes = load_network(network_file)
    # Normalization by MinMaxScaler
    scaler = preprocessing.MinMaxScaler()
    feat_raw = scaler.fit_transform(np.abs(feat_dic[feat_name_lst[0]].reindex(net_nodes['nodes'].values.tolist(), fill_value=0)))
    # Concatenate multiple features to form one feature matrix
    if len(feat_file_lst) > 1:
        for i in range(1,len(feat_file_lst)):
            feat_raw = np.concatenate((feat_raw, scaler.fit_transform(np.abs(feat_dic[feat_name_lst[i]].reindex(net_nodes['nodes'].values.tolist(), fill_value=0)))), axis=1)

    return pd.DataFrame(feat_raw,index=net_nodes['nodes'].values.tolist(),columns=feat_name_lst)

def create_edge_index(network_file,net_features):
    """
    Convert the edges in a network into edges indexed by integer ids, which is necessary to build an object typeof torch_geometric.data.Data.
    :param network_file: Full pathname of the network file
    :param net_features (class: pandas.DataFrame): Concatenated feature matrix with n rows(genes) and m columns(features)
    :return (class: pandas.DataFrame): Edges indexed by integer ids
    """
    net, _ = load_network(network_file)
    node_df = pd.DataFrame({'name':net_features.index.values.tolist(),
                            'id':[i for i in np.arange(0,net_features.shape[0])]})
    net = pd.merge(left=net,right=node_df,how='left',left_on='source',right_on='name')
    net.columns=['source','target','sourcename','sourceid']
    net = pd.merge(left=net, right=node_df, how='left',left_on='target',right_on='name')
    net.columns=['source','target','sourcename','sourceid','targetname','targetid']
    edge_index1 = net.loc[:,['sourceid','targetid']]
    # Treat the graph as undirected graph
    edge_index2 = net.loc[:,['targetid','sourceid']]
    edge_index = pd.concat([edge_index1,edge_index2],axis=0)
    return edge_index

def generate_5CV_set(drivers,nondrivers,randseed):
    """
    Generate 5CV splits.
    :param drivers: List of canonical driver genes(positive samples)
    :param nondrivers: List of nondriver genes(negative samples)
    :param randseed: Random seed
    :return: 5CV splits sorted in a dictionary
    """
    # StratifiedKFold
    X, y = drivers + nondrivers, np.hstack(([1]*len(drivers), [0]*len(nondrivers)))
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=randseed)
    X_5CV = {}
    cv_idx=1
    for train, test in skf.split(X, y):
        # train/test sorts the sample indices in X list.
        # For each split, we should convert the indices in train/test to names
        train_set=[]
        train_label=[]
        test_set=[]
        test_label=[]
        for i in train:
            train_set.append(X[i])
            train_label.append(y[i])
        for i in test:
            test_set.append(X[i])
            test_label.append(y[i])
        X_5CV['train_%d' % cv_idx] = train_set
        X_5CV['test_%d' % cv_idx] = test_set
        X_5CV['train_label_%d' % cv_idx] = train_label
        X_5CV['test_label_%d' % cv_idx] = test_label
        cv_idx = cv_idx + 1
    return X_5CV

feat_file_lst = ['./preprocess_data/gene_mutation/MF_BLCA_mutation_matrix.tsv',
                 './preprocess_data/gene_expression/GE_BLCA_expression_matrix.tsv',
                 './preprocess_data/DNA_methylation/METH_BLCA_methylation_RATIO_mean.tsv']

network_file = './data/PathNet/PathNet.txt'

feat_name_lst = ['mut','exp','methy']


# Concatenate multiple features to form one feature matrix
net_features = build_customized_feature_matrix(feat_file_lst, network_file, feat_name_lst)

# A dataset contains the following data:
# feature: the gene feature matrix
# edge_index: graph edges for training model
# node_name: gene names
# feature_name: feature names
# label: True labels of genes (0 for negative samples and 1 for positive samples),
# k_sets: 5CV splits that randomly generated for ten times
# mask: mask for training a single model without cross-validation
dataset=dict()
dataset['feature'] = torch.FloatTensor(np.array(net_features))
dataset['node_name'] = net_features.index.values.tolist()
# Create edge_index by edges in network file
edge_index = create_edge_index(network_file,net_features)
dataset['edge_index'] = torch.LongTensor(np.array(edge_index).transpose())
dataset['feature_name'] = net_features.columns.values.tolist()

# Generate 10 rounds 5CV splits

# Canonical driver genes (positive samples)
d_lst = pd.read_table(filepath_or_buffer='./data/796_drivers.txt', sep='\t', header=None, index_col=None,
                      names=['driver'])
d_lst = d_lst['driver'].values.tolist()

# Nondriver genes (negative samples)
nd_lst = pd.read_table(filepath_or_buffer='./data/2187_nondrivers.txt', sep='\t', header=None,
                       index_col=None, names=['nondriver'])
nd_lst = nd_lst['nondriver'].values.tolist()

# True labels of genes
labels = []
mask = [] # mask for training a single model without cross-validation
for g in dataset['node_name']:
    if g in d_lst:
        labels.append(1)
    else:
        labels.append(0)
    if (g in d_lst) or (g in nd_lst):
        mask.append(True)
    else:
        mask.append(False)

d_in_net = [] # Canonical driver genes in the network
nd_in_net = [] # Nondriver genes in the network
for g in dataset['node_name']:
    if g in d_lst:
        d_in_net.append(g)
    elif g in nd_lst:
        nd_in_net.append(g)

k_sets_net = dict()
for k in np.arange(0,10): # Randomly generate 5CV splits for ten times
    k_sets_net[k] = []
    randseed = (k+1)%100+(k+1)*5
    cv = generate_5CV_set(d_in_net,nd_in_net,randseed)
    for cv_idx in np.arange(1,6):
        tr_mask = [] # train mask
        te_mask = [] # test mask
        for g in dataset['node_name']:
            if g in cv['train_%d' % cv_idx]:
                tr_mask.append(True)
            else:
                tr_mask.append(False)
            if g in cv['test_%d' % cv_idx]:
                te_mask.append(True)
            else:
                te_mask.append(False)
        tr_mask = np.array(tr_mask)
        te_mask = np.array(te_mask)
        k_sets_net[k].append((tr_mask,te_mask))


dataset['label'] = torch.FloatTensor(np.array(labels))
dataset['split_set'] = k_sets_net
dataset['mask'] = np.array(mask)
# Save the dataset as pickle file, which can be used for training HGDC
with open('./data/dataset_BLCA_ten_5CV.pkl', 'wb') as f:
    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
