import pickle
import torch
import numpy as np
import pandas as pd
import networkx as nx

def load_obj( name ):
    with open( name , 'rb') as f:
        return pickle.load(f)

def load_net_specific_dataset_ten_5CV(net_name):
    dataset = load_obj('./data/dataset_%s_ten_5CV.pkl' % net_name)
    features = dataset['feature']
    edge_index = dataset['edge_index']
    feature_names = dataset['feature_name']
    node_names = dataset['node_name']
    labels = dataset['label']
    k_sets = dataset['split_set']
    return features, edge_index, feature_names, node_names, labels, k_sets
