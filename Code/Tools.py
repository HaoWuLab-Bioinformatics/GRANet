import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_dir, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.

                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.

                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.

                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.save_dir = save_dir
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.

        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')
        torch.save(model.state_dict(), self.save_dir + '.pkl')
        self.val_loss_min = val_loss

class SavaBestModel:

    def __init__(self,save_dir):

        self.save_dir = save_dir
        self.best_score = None

    def __call__(self, auroc, auprc, model):

        score = 0.8 * auroc + 0.2 * auprc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)

        elif score < self.best_score:
            pass

        else:
            self.best_score = score
            self.save_checkpoint(model)

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.save_dir+'/best_model.pkl')



def progress_bar(finish_tasks_number, tasks_number):
    """


    :param finish_tasks_number: int,
    :param tasks_number: int,
    :return:
    """

    percentage = round(finish_tasks_number / tasks_number * 100)
    print("\rprocess: {}%: ".format(percentage), "▓" * (percentage // 2), end="")
    sys.stdout.flush()


def Evaluation(y_true, y_pred,flag=False):
    if flag:
        y_p = y_pred[:,-1]
        y_p = y_p.cpu().detach().numpy()
        y_p = y_p.flatten()
    else:
        y_p = y_pred.cpu().detach().numpy()
        y_p = y_p.flatten()

    y_t = y_true.cpu().numpy().flatten().astype(int)
    AUC = roc_auc_score(y_true=y_t, y_score=y_p)
    AUPR = average_precision_score(y_true=y_t,y_score=y_p)
    return AUC, AUPR, #ACC



if __name__ == '__main__':
    for i in range(0, 101):
        progress_bar(i, 100)
        time.sleep(0.05)

