import argparse
import torch
import numpy as np
from sklearn import metrics
from data.data_loader import load_net_specific_data
from sklearn import svm

parser = argparse.ArgumentParser()
parser.add_argument('-graph_diffusion', type=str, default='none', help='Graph diffusion name, options:[none].')
parser.add_argument('-is_5_CV_test', type=bool, default=True, help='Run 5-CV test.')# When setting is_5_CV_test=True, make sure the pkl file include masks of different 5CV splits.
parser.add_argument('-dataset_file', type=str, default='./data/PPNet/dataset_PPNet_ten_5CV.pkl', help='The path of the input pkl file.')
args = parser.parse_args(args=[])

data = load_net_specific_data(args)

X = data.x.numpy()
Y = data.y.numpy()

AUC = np.zeros(shape=(10, 5))
AUPR = np.zeros(shape=(10, 5))

for r in range(10):
    for cv_run in range(5):
        tr_mask, te_mask = data.mask[r][cv_run]
        # training process
        svc = svm.SVC(kernel='rbf',probability=True).fit(X[tr_mask, :], Y[tr_mask])
        # test process
        y_te = Y[te_mask]
        x_te = X[te_mask]
        pred = svc.predict(x_te)
        precision, recall, _thresholds = metrics.precision_recall_curve(y_te,pred)
        aupr_te = metrics.auc(recall, precision)
        auc_te = metrics.roc_auc_score(y_te,pred)
        AUC[r][cv_run], AUPR[r][cv_run] = auc_te, aupr_te

    print('Round--%d  AUC: %.5f, AUPR: %.5f' % (r, np.mean(AUC[r, :]), np.mean(AUPR[r, :])))



