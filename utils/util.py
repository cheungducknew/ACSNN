import dgl
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt
from torch.nn import functional as F

# Add regression metrics
from sklearn.metrics import mean_squared_error, r2_score
try:
    from scipy.stats import pearsonr, spearmanr
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def plotROC(y, z, pstr=''):
    fpr, tpr, tt = metrics.roc_curve(y, z)
    roc_auc = roc_auc_score(y, z)
    plt.figure()
    plt.plot(fpr, tpr, 'o-')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.grid()
    plt.title('ROC ' + pstr + ' AUC: '+str(roc_auc_score(y, z)))


# keep existing classification helpers if desired
def evaluate_metrics(y, y_pred, y_proba, draw_roc=False):

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    ba = balanced_accuracy_score(y, y_pred)
    tpr = recall_score(y, y_pred)
    tnr = tn/(tn+fp)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    
    if draw_roc:
        plotROC(y, y_pred)
    
    return tn, fp, fn, tp, round(ba, 3), round(tpr, 3), round(tnr, 3), round(f1, 3), round(auc, 3)


# ---------- modified predict for regression ----------
def predict(args, model, data_loader, criterion, optimizer, is_train):

    device = args['DEVICE']

    total_loss = 0.0
    output_total, y_total = [], []

    for i, X_data, y_data in data_loader:
        # prepare y_data as float tensor (data_loader already returns float tensors)
        y_data = torch.from_numpy(np.array(y_data)).float() if not isinstance(y_data, torch.Tensor) else y_data.float()

        if args['MODEL'] == 'acgcn-mmp' or args['MODEL'] == 'acgcn-snn':
            smiles1 = [x[0]['GRAPH_SMILES1'] for x in X_data]
            smiles2 = [x[0]['GRAPH_SMILES2'] for x in X_data]

            batch_smiles1 = dgl.batch(smiles1)
            batch_smiles2 = dgl.batch(smiles2)

            if torch.cuda.is_available():
                batch_smiles1 = batch_smiles1.to(device)
                batch_smiles2 = batch_smiles2.to(device)
                y_data = y_data.to(device)

            outputs = model(batch_smiles1, batch_smiles2)

        elif args['MODEL'] == 'acgcn-sub':
            core = [x[0]['GRAPH_CORE'] for x in X_data]
            sub1 = [x[0]['GRAPH_SUB1'] for x in X_data]
            sub2 = [x[0]['GRAPH_SUB2'] for x in X_data]

            batch_core = dgl.batch(core)
            batch_sub1 = dgl.batch(sub1)
            batch_sub2 = dgl.batch(sub2)

            if torch.cuda.is_available():
                batch_core = batch_core.to(device)
                batch_sub1 = batch_sub1.to(device)
                batch_sub2 = batch_sub2.to(device)
                y_data = y_data.to(device)

            outputs = model(batch_core, batch_sub1, batch_sub2)

        loss = criterion(outputs, y_data)
        output_total += outputs.detach().cpu().tolist()

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        y_total += y_data.detach().cpu().tolist()

    return model, loss, total_loss, output_total, y_total


def get_actual_label(data_loader):
    
    y_arr = []
    for i, X_data, y_data in data_loader:
        # y_data might be torch tensor
        if isinstance(y_data, torch.Tensor):
            y_arr += y_data.detach().cpu().tolist()
        else:
            y_arr += list(y_data)
    
    return y_arr


class WeightedBCELoss(torch.nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights
        self.eps = 1e-9

    def forward(self, output, target):
        if self.weights is not None:
            assert len(self.weights) == 2
            loss = self.weights[1] * (target * torch.log(output + self.eps)) + \
                self.weights[0] * ((1 - target) * torch.log(1 - output + self.eps))
        else:
            loss = target * torch.log(output + self.eps) + (1 - target) * torch.log(1 - output + self.eps)
            print(output, target)
            print(loss)
        return torch.neg(torch.mean(loss))


# ---------- new regression metrics printing ----------
def print_regression_metrics(y_pred, y_actual):
    """
    y_pred, y_actual: lists or numpy arrays (floats)
    prints MSE, RMSE, R2, Pearson (PCC), Spearman
    """
    y_pred = np.array(y_pred)
    y_actual = np.array(y_actual)

    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual, y_pred)

    # Pearson correlation
    if _HAS_SCIPY:
        try:
            pcc, _ = pearsonr(y_actual, y_pred)
        except Exception:
            pcc = np.corrcoef(y_actual, y_pred)[0,1]
        try:
            spearman_corr, _ = spearmanr(y_actual, y_pred)
        except Exception:
            # fallback to pandas
            spearman_corr = pd.Series(y_actual).corr(pd.Series(y_pred), method='spearman')
    else:
        # fallback to numpy/pandas
        pcc = np.corrcoef(y_actual, y_pred)[0,1]
        spearman_corr = pd.Series(y_actual).corr(pd.Series(y_pred), method='spearman')

    print("============== Regression Performance =================")
    print("* MSE : {:.6f}".format(mse))
    print("* RMSE: {:.6f}".format(rmse))
    print("* R2  : {:.6f}".format(r2))
    print("* PCC : {:.6f}".format(pcc))
    print("* Spearman: {:.6f}".format(spearman_corr))
    print("=======================================================")