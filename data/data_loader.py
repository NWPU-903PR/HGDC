import pickle
import torch
from utils.auxiliary_graph_generator import generate_auxiliary_graph
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

def load_obj( name ):
    """
    Load dataset from pickle file.
    :param name: Full pathname of the pickle file
    :return: Dataset type of dictionary
    """
    with open( name , 'rb') as f:
        return pickle.load(f)


# When setting is_5_CV_test=True, make sure the pkl file include masks of different 5CV splits.
# args.dataset_file='./data/GGNet/dataset_GGNet_ten_5CV.pkl'
# args.is_5_CV_test = Ture

def load_net_specific_data(args):
    """
    Load network-specific dataset from the pickle file.
    :param args: Arguments received from command line
    :return: Data for training model (class: 'torch_geometric.data.Data')
    """
    dataset = load_obj(args.dataset_file)

    std = StandardScaler()
    features = std.fit_transform(dataset['feature'].detach().numpy())
    features = torch.FloatTensor(features)

    if args.is_5_CV_test:
        mask = dataset['split_set']
    else:
        mask = dataset['mask']

    if args.graph_diffusion == 'none':
        data = Data(x=features, y=dataset['label'], edge_index=dataset['edge_index'], mask=mask, node_names=dataset['node_name'])
        return data
    else:
        edge_index_aux = generate_auxiliary_graph(args, dataset)
        data = Data(x=features, y=dataset['label'], edge_index=dataset['edge_index'], edge_index_aux=edge_index_aux, mask=mask, node_names=dataset['node_name'])
        return data










