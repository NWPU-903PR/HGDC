import argparse
import torch
import numpy as np
from sklearn import metrics
import torch.nn.functional as F
from models.baselines.MTGCN import MTGCNNet
from models.baselines.EMOGI import EMOGINet
from models.baselines.Chebnet import ChebnetNet
from models.baselines.GAT import GATNet
from models.baselines.GCN import GCNNet
from data.data_loader import load_net_specific_data
from sklearn import svm

parser = argparse.ArgumentParser()
parser.add_argument('--graph_diffusion', type=str, default='none', help='Graph diffusion name, options:[none].')
parser.add_argument('--is_5_CV_test', type=bool, default=True, help='Run 5-CV test.')# When setting is_5_CV_test=True, make sure the pkl file include masks of different 5CV splits.
parser.add_argument('--dataset_file', type=str, default='./data/PPNet/dataset_PPNet_ten_5CV.pkl', help='The path of the input pkl file.')
parser.add_argument('--model', type=str, default='EMOGI', help='Model name, options:[MTGCN, EMOGI, Chebnet, GAT, GCN, SVM].')
parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--device', type=int, default=0, help='The id of GPU.')
args = parser.parse_args()

device = torch.device('cuda:%d' % args.device if torch.cuda.is_available() else 'cpu')

model_dict = {
              'MTGCN':MTGCNNet,
              'EMOGI':EMOGINet,
              'Chebnet':ChebnetNet,
              'GAT':GATNet,
              'GCN':GCNNet
             }


data = load_net_specific_data(args)
data = data.to(device)

@torch.no_grad()
def test(data,model_name,mask):
    model.eval()
    if model_name == 'MTGCN':
        x, _, _, _ = model(data)
    else:
        x = model(data)
    pred = torch.sigmoid(x[mask])

    precision, recall, _thresholds = metrics.precision_recall_curve(data.y[mask].cpu().numpy(),
                                                                    pred.cpu().detach().numpy())
    area = metrics.auc(recall, precision)

    return metrics.roc_auc_score(data.y[mask].cpu().numpy(), pred.cpu().detach().numpy()), area

# Ten times for 5_CV
AUC = np.zeros(shape=(10, 5))
AUPR = np.zeros(shape=(10, 5))

for i in range(10):
    for cv_run in range(5):
        tr_mask, te_mask = data.mask[i][cv_run]
        # Train GNN-based methods, such as EMOGI, MTGCN, GAT, GCN and Chebnet.
        if args.model != 'SVM':
            model = model_dict[args.model](args).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            for epoch in range(1, args.epochs + 1):
                # Training model
                model.train()
                optimizer.zero_grad()
                if args.model == 'MTGCN':
                    pred, rl, c1, c2 = model(data)
                    loss = F.binary_cross_entropy_with_logits(pred[tr_mask], data.y[tr_mask].view(-1, 1)) / (c1 * c1) + rl / (
                                c2 * c2) + 2 * torch.log(c2 * c1)
                elif args.model =='EMOGI':
                    pred = model(data)
                    loss = F.binary_cross_entropy_with_logits(pred[tr_mask], data.y[tr_mask].view(-1, 1),
                                                              pos_weight=torch.tensor([45]).to(device))
                else:
                    pred = model(data)
                    loss = F.binary_cross_entropy_with_logits(pred[tr_mask], data.y[tr_mask].view(-1, 1))
                loss.backward()
                optimizer.step()

                if epoch % 100 == 0:
                    print(f'Training epoch: {epoch:03d}')
            AUC[i][cv_run], AUPR[i][cv_run] = test(data, args.model, te_mask)
        # Train SVM model
        else:
            # training process
            X = data.x.cpu().numpy()
            Y = data.y.cpu().numpy()
            svc = svm.SVC(kernel='rbf', probability=True).fit(X[tr_mask, :], Y[tr_mask])
            # test process
            y_te = Y[te_mask]
            x_te = X[te_mask]
            pred = svc.predict(x_te)
            precision, recall, _thresholds = metrics.precision_recall_curve(y_te, pred)
            aupr_te = metrics.auc(recall, precision)
            auc_te = metrics.roc_auc_score(y_te, pred)
            AUC[i][cv_run], AUPR[i][cv_run] = auc_te, aupr_te

        print('Round--%d CV--%d  AUC: %.5f, AUPR: %.5f' % (i, cv_run + 1, AUC[i][cv_run], AUPR[i][cv_run]))

    print('Round--%d Mean AUC: %.5f, Mean AUPR: %.5f' % (i, np.mean(AUC[i, :]), np.mean(AUPR[i, :])))


print('%s 10 rounds for 5CV-- Mean AUC: %.4f, Mean AUPR: %.4f' % (args.model, AUC.mean(), AUPR.mean()))