import math
import networkx as nx
import numpy as np
import torch

import os
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
__all__ = ['FCMuonsGPU']

class FCMuonsGPU(object):

    def __init__(self, name, sub):
        super(FCMuonsGPU, self).__init__()

        self.all=torch.load(name)
        #self.shape=self.all['x']
        #self.adj=self.all['edge_index']
    
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self,idx):

        return self.all[idx]

