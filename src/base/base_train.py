import tensorflow as tf
import pdb, time
from evaluator import Evaluator
from HGNN import STGCN
from torch import nn
import torch
import torch.optim as optim
from hypergraph_utils import generate_G_from_H
import pprint as pp
import numpy as np
device = 'cuda'
class BaseTrain:
    def __init__(self, data, config):
        self.config = config
        self.data = data
        self.H = np.load('hypergraph.npy').astype(np.float32)
        self.G = generate_G_from_H(self.H)
        self.G = torch.FloatTensor(np.array(self.G)).to(device)
        self.hgnn_model =  STGCN(423, 1, 50,3).to(device)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.evaluator = Evaluator(config)
    def train(self):
        l_r = 2e-4
        optimizer = optim.Adam(self.hgnn_model.parameters(), 
                       lr=l_r, 
                       weight_decay=5e-4)
        for cur_epoch in range(300):
            self.hgnn_model.train()
            loss, report_all, report_topk = self.train_epoch(optimizer)
            self.hgnn_model.eval()
            self.evaluator.evaluate(self.hgnn_model,self.data,'test',self.G)
                

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
