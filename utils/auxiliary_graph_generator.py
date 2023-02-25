from torch_geometric.data import Data
import torch_geometric.transforms as T
import networkx as nx
import pandas as pd
import numpy as np
import torch

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

def create_network(genes,edges,direction=False):
    """
    Create a networkx graph object by genes and edges.
    :param genes (class: pandas.DataFrame): List of genes
    :param edges (class: pandas.DataFrame): List of edges
    :param direction (bool, optional): Whether the graph object is directed or undirected
    :return: A networkx graph object
    """
    gene_list=genes.iloc[:,0].values.tolist()
    n=len(gene_list)
    if direction:
        G=nx.DiGraph()
    else :
        G=nx.Graph()

    for i in np.arange(0,n):
        G.add_node(gene_list[i])

    for _,row in edges.iterrows():
        G.add_edge(row['source'],row['target'],weight=1)
    return G

def sparse_dense_graph(graph_adj, topN=30):
    """
    Sparse the dense graph by retaining the topN edges for each node
    :param graph_adj: Adjacency matrix of dense graph
    :param topN (int, optional): Number of edges to retain for each node
    :return (class: numpy.array): Adjacency matrix of sparse graph
    """
    net_mtx = np.zeros((graph_adj.shape[0],graph_adj.shape[1]))
    indices = np.argsort(-graph_adj)[:,:topN]
    for i in np.arange(0, indices.shape[0]):
        for j in np.arange(0, indices.shape[1]):
            net_mtx[i, indices[i,j]] = graph_adj[i,indices[i,j]]
    return net_mtx

def convert_adj_to_edgeset(adj, node_df):
    """
    Extract edges from adjacency matrix of graph.
    :param adj: Adjacency matrix of graph
    :param node_df (class: pandas.DataFrame): node names
    :return (class: pandas.DataFrame): Edges in the graph
    """
    sour_lst = []
    targ_lst = []
    for i in np.arange(0, node_df.shape[0]):
        row = adj[i,:]
        targ = node_df.loc[row.nonzero()[0], :]['nodes'].tolist()
        sour = [node_df.iloc[i, 0] for j in range(0, len(targ))]
        targ_lst = targ_lst + targ
        sour_lst = sour_lst + sour
    return pd.DataFrame({'source':sour_lst, 'target':targ_lst})


def generate_auxiliary_graph(args, dataset):
    """
    Construct the auxiliary graph based on the original graph.
    :param args: Arguments received from command line
    :param dataset (class: dictionary): Dataset loaded from a pickle file
    :return (class: torch.LongTensor): edges indexed by ids
    """
    data = Data(x=dataset['feature'], y=dataset['label'], edge_index=dataset['edge_index'])
    if args.graph_diffusion == 'ppr':
        gdc = T.GDC(self_loop_weight=None, normalization_in='sym',
                    normalization_out='col',
                    diffusion_kwargs=dict(method=args.graph_diffusion, alpha=args.ppr_alpha, eps=args.ppr_eps),
                    sparsification_kwargs=dict(method='threshold', avg_degree=args.net_avg_deg),
                    exact=True)
        data = gdc(data)
        print('Auxiliary graph of %s is finished...' % args.graph_diffusion)
        return data.edge_index

    elif args.graph_diffusion == 'heat':
        gdc = T.GDC(self_loop_weight=None, normalization_in='sym',
                    normalization_out='col',
                    diffusion_kwargs=dict(method=args.graph_diffusion, t=args.hk_t),
                    sparsification_kwargs=dict(method='threshold', avg_degree=args.net_avg_deg),
                    exact=True)
        data = gdc(data)
        print('Auxiliary graph of %s is finished...' % args.graph_diffusion)
        return data.edge_index

    elif args.graph_diffusion == 'simrank':
        # SimRank may cost tens of minutes...
        print('The computation of SimRank may take about tens of minutes......')
        net, nodes = load_network(args.net_file)
        G = create_network(genes = nodes, edges = net, direction = False)
        # Calculate simrank values of all the pairs of nodes in the graph G
        res_simrank = nx.simrank_similarity(G)
        # Obtain the adjacent matrix of simrank graph based on res_simrank
        simrank_adj = np.array([[res_simrank[u][v] for v in G] for u in G])
        # Remove self-loops from simrank graph
        for i in np.arange(0,simrank_adj.shape[0]):
            simrank_adj[i,i] = 0
        # Sparse the dense simrank graph by retaining the topN similar nodes for each node.
        simrank_adj_spr = sparse_dense_graph(simrank_adj)

        # Construct the edge_index tensor from the sparse simrank graph
        edge_set = convert_adj_to_edgeset(simrank_adj_spr, nodes)
        # Convert names of edges in edge_set to idx of edges according to the order of nodes in list of dataset['node_name']
        node_df = pd.DataFrame({'name':dataset['node_name'],'id':[i for i in np.arange(0, len(dataset['node_name']))]})
        edge_set = pd.merge(left = edge_set, right = node_df, how = 'left', left_on = 'source', right_on = 'name')
        edge_set.columns = ['source', 'target', 'sourcename', 'sourceid']
        edge_set = pd.merge(left = edge_set, right = node_df, how = 'left', left_on = 'target', right_on = 'name')
        edge_set.columns = ['source', 'target', 'sourcename', 'sourceid', 'targetname', 'targetid']
        edge_index = edge_set.loc[:,['sourceid','targetid']]
        edge_index = torch.LongTensor(np.array(edge_index).transpose())
        print('Auxiliary graph of %s is finished...' % args.graph_diffusion)
        return edge_index




