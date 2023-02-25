import argparse
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from models.HGDC import HGDCNet
from data.data_loader import load_net_specific_data
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--graph_diffusion', type=str, default='ppr', help='Graph diffusion name, options:[ppr, heat, simrank, ].')
parser.add_argument('--ppr_alpha', type=float, default=0.9, help='Return probability in PPR.')
parser.add_argument('--ppr_eps', type=float, default=0.0001, help='Threshold to bound edges at in PPR.')
parser.add_argument('--net_avg_deg', type=int, default=50, help='Average node degree of a network.') # GGNet 111, PathNet 24, PPNet 50
parser.add_argument('--hk_t', type=int, default=1, help='Times of diffusion in Heat kernel PageRank.')
parser.add_argument('--is_5_CV_test', action='store_true', help='Run 5-CV test.')
parser.add_argument('--dataset_file', type=str, default='./data/PPNet/dataset_PPNet.pkl', help='The path of the input pkl file.') # When setting is_5_CV_test=True, make sure the pkl file include masks of different 5CV splits.
parser.add_argument('--net_file', type=str, default='./data/PPNet/PPNet.txt', help='The path of network file, corresponding to dataset_file.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--w_decay', type=float, default=0.00001, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--in_channels', type=int, default=58, help='Dimension of node features.')
parser.add_argument('--hidden_channels', type=int, default=100, help='Dimension of hidden Linear layers.')
parser.add_argument('--device', type=int, default=0, help='The id of GPU.')
parser.add_argument('--times', type=int, default=100, help='Number of times to repeat training HGDC model.')
args = parser.parse_args()

device = torch.device('cuda:%d' % args.device if torch.cuda.is_available() else 'cpu')
# Load dataset from pkl file
data = load_net_specific_data(args)
data = data.to(device)
pred_all = np.zeros((data.num_nodes, 1))
# Repeat training HGDC for args.times times
for i in np.arange(0, args.times):
    # Creating model
    model = HGDCNet(args).to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.linear1.parameters(), weight_decay=args.w_decay),
        dict(params=model.linear_r0.parameters(), weight_decay=args.w_decay),
        dict(params=model.linear_r1.parameters(), weight_decay=args.w_decay),
        dict(params=model.linear_r2.parameters(), weight_decay=args.w_decay),
        dict(params=model.linear_r3.parameters(), weight_decay=args.w_decay),
        dict(params=model.weight_r0, lr=args.lr * 0.1),
        dict(params=model.weight_r1, lr=args.lr * 0.1),
        dict(params=model.weight_r2, lr=args.lr * 0.1),
        dict(params=model.weight_r3, lr=args.lr * 0.1)
    ], lr=args.lr)

    # Training model
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(data)
        loss = F.binary_cross_entropy_with_logits(pred[data.mask], data.y[data.mask].view(-1, 1))
        loss.backward()
        optimizer.step()
    print('Training HGDC for %d times.' % (i + 1))
    # Predicting via trained model
    pred = model(data)
    pred_all = pred.cpu().detach().numpy() + pred_all

pred_all = pred_all/args.times
pre_res = pd.DataFrame(pred_all,columns=['score'],index=data.node_names)
pre_res.sort_values(by=['score'], inplace=True, ascending=False)
# Save the final ranking list of predicted driver genes
pre_res.to_csv(path_or_buf='./predicted_socres_HGDC.txt', sep='\t', index=True, header=True)



