from base.base_train import BaseTrain
import numpy as np
import time, random
import os
import time
import copy
import torch
import torch.optim as optim
import pprint as pp
from hypergraph_utils import generate_G_from_H
from HGNN import HGNN
from torch import nn
from tqdm import tqdm
device='cuda'

class Trainer(BaseTrain):
    def __init__(self, data, config):
        super(Trainer, self).__init__(data, config)
        self.keep_prob = 1-config.dropout
        
    def train_epoch(self, optimizer):
        all_x, all_y, all_rt = next(self.data.get_batch('train', self.config.lookback))
        for x, y, rt in tqdm(zip(all_x, all_y, all_rt)):
            optimizer.zero_grad()
            train_price = torch.tensor(x, dtype = torch.float32).to(device)
            train_label = torch.LongTensor(y).to(device)
            output = self.hgnn_model(self.G,train_price.reshape(1,423,50,1))
            loss = self.cross_entropy(output, torch.argmax(train_label,1))
            loss.backward()
            optimizer.step()
        return 1, 1,1


