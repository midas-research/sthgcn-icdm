import math
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score
import torch
import time
import datetime  

import torch.nn.functional as F
import pickle
device = 'cuda'
class Evaluator:
    def __init__(self, config):
        self.config = config
        self.n_labels = len(config.label_proportion)



    def evaluate(self, model,data, phase,G):
        all_x, all_y, all_rt = next(data.get_batch(phase, self.config.lookback))
        for x, y, rt in zip(all_x, all_y, all_rt):
            train_price = torch.tensor(x, dtype = torch.float32).to(device)
            train_label = torch.LongTensor(y).to(device)
            output = model(G,train_price.reshape((1,423,50,1)))
            pred = torch.argmax(output,1).detach().cpu().numpy()
            label = np.argmax(y,1)
            prob = F.softmax(output, dim=1).detach().cpu().numpy()
            pred_li.append(pred)
            true_li.append(label)
            metrics_all, metrics_topk = self.metric(label, pred, prob, rt)
        true_li1 = true_li
        pred_li1 = pred_li
        true_li = np.array(true_li).reshape(-1).tolist()
        pred_li = np.array(pred_li).reshape(-1).tolist()
        print(classification_report(true_li, pred_li))
        return 1, 1, 1


    def create_confusion_matrix(self, y, y_, is_distribution=False):
        n_samples = float(y_.shape[0])   # get dimension list
        if is_distribution:
            label_ref = np.argmax(y_, 1)  # 1-d array of 0 and 1
            label_hyp = np.argmax(y, 1)
        else:
            label_ref, label_hyp = y, y_

        # p & n in prediction
        p_in_hyp = np.sum(label_hyp)
        n_in_hyp = n_samples - p_in_hyp

        # Positive class: up
        tp = np.sum(np.multiply(label_ref, label_hyp))  # element-wise, both 1 can remain
        fp = p_in_hyp - tp  # predicted positive, but false

        # Negative class: down
        tn = n_samples - np.count_nonzero(label_ref + label_hyp)  # both 0 can remain
        fn = n_in_hyp - tn  # predicted negative, but false
        return float(tp), float(fp), float(tn), float(fn)

    def get_mcc(self, tp, fp, tn, fn):
        core_de = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        return (tp * tn - fp * fn) / math.sqrt(core_de) if core_de else 0

    def get_f1(self, tp, fp, tn, fn):
        eps = 1e-10
        precision = tp / (tp+fp+eps)
        recall = tp / (tp+fn+eps)
        return 2 * (precision*recall) / (precision+recall+eps)

    def get_f1(self, y, y_):
        # y : label | y_ : pred
        return f1_score(y,y_,average='macro'), f1_score(y,y_,average='micro')

    def get_acc(self, conf_mat):
        accuracy = conf_mat.trace()/conf_mat.sum()
        if self.n_labels==2:
            compact_accuracy = accuracy
        else:
            # It is actually a recall of up down
            compact_conf_mat = np.take(conf_mat,[[0,2],[6,8]])
            compact_accuracy = compact_conf_mat.trace()/compact_conf_mat.sum()
        return accuracy, compact_accuracy

    def expected_return(self, pred, prob, returns):
        # To create neuralized portfolio
        n_mid = prob.shape[0]//2
        # sorted : ascending order (based on down probabilty)
        # both side have exactly the half size of the universe
        short_half_idx = np.argsort(prob[:,0])[-n_mid:]
        long_half_idx = np.argsort(prob[:,-1])[-n_mid:]
        # if prediction was neutral, we don'y count it as our return
        short_rts = (returns[short_half_idx]*(pred[short_half_idx]==0)).mean() * (-1)
        long_rts = (returns[long_half_idx]*(pred[long_half_idx]==(self.n_labels-1))).mean()
        return (short_rts + long_rts) * 100

    def filter_topk(self, label, pred, prob, returns, topk):
        short_k_idx = np.argsort(prob[:,0])[-topk:]
        long_k_idx = np.argsort(prob[:,-1])[-topk:]
        topk_idx = np.concatenate([short_k_idx, long_k_idx])
        return label[topk_idx], pred[topk_idx], prob[topk_idx], returns[topk_idx]

    def cal_metric(self, label, pred, prob, returns):
        exp_returns = self.expected_return(pred, prob, returns)
        conf_mat = confusion_matrix(label, pred, labels=[i for i in range(self.n_labels)])
        acc, cpt_acc = self.get_acc(conf_mat)
        mac_f1, mic_f1 = self.get_f1(label, pred)
        pred_rate = [(pred==i).sum()/pred.shape[0] for i in range(self.n_labels)]
        return pred_rate, (acc, cpt_acc), mac_f1, mic_f1, exp_returns

    def metric(self, label, pred, prob, returns, topk=30):
        metric_all = self.cal_metric(label, pred, prob, returns)
        label, pred, prob, returns = self.filter_topk(label, pred, prob, returns, topk)
        metric_topk = self.cal_metric(label, pred, prob, returns)
        return metric_all, metric_topk

