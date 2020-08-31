#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 17:51:58 2020

@author: alihariri
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 09:59:05 2020

@author: alihariri
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
#from layers import GraphConvolution
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
import torch_geometric
import torch_geometric.nn as tnn
from GCN_Messaging import GCNConv_Message


class GCN_Message(nn.Module):
    def __init__(self,in_channels, out_channels1, out_channels2,out_channels3,deco2,deco3,deco4,num_layers, dropout,batch_size):
        super(GCN_Message, self).__init__()   

        

        self.dec1=nn.Linear(64,deco2)
        self.dec2=tnn.SAGEConv(deco2,deco3)
        self.dec3=tnn.SAGEConv(deco3,deco4)
        self.dec4=tnn.SAGEConv(deco4,3)
        """
        self.batch_size=batch_size
        self.out_channels2=out_channels2
        self.lin3=nn.Linear(1,20)
        self.lin32=nn.Linear(20,4)
        self.bn1 = nn.BatchNorm1d(num_features=deco2)
        self.bn2=nn.BatchNorm1d(num_features=deco3)
        """
        """
        Gated GNN
        """
        self.ggn=tnn.SAGEConv(in_channels,out_channels1)
        self.ggn2=tnn.SAGEConv(out_channels1,out_channels2)
        self.ggn3=tnn.SAGEConv(out_channels2,out_channels3)

   
        self.tr2=nn.Linear(out_channels3,64)
  
        
        

    def encode(self,x,adj):
        hidden1=self.ggn(x,adj)
        hidden1=F.tanh(hidden1)
        hidden1=F.dropout(hidden1)
        hidden1=self.ggn2(hidden1,adj)
        hidden1=F.tanh(hidden1)
        hidden1=F.dropout(hidden1)
        hidden1=self.ggn3(hidden1,adj)
        hidden1=F.tanh(hidden1)
        hidden1=F.dropout(hidden1)
        #hidden2=self.tr1(hidden1)
        #hidden2=self.ggn(trans1,adj)
        return self.tr2(hidden1),self.tr2(hidden1) #hid2

    def reparametrize(self, mu, logvar):
      if self.training:
          return mu + torch.randn_like(logvar) * torch.exp(logvar)
      else:
          return mu  

    def decode(self,z,adj):
        out1=self.dec1(z)
        out1=F.leaky_relu(out1)
        out1=F.dropout(out1)
        #out1=self.bn1(out1)
        out2=self.dec2(out1,adj)
        out2=F.leaky_relu(out2)
        out2=F.dropout(out2)
        #out2=self.bn2(out2)
        out2=self.dec3(out2,adj)
        out2=F.tanh(out2) ##har: tanh
        out2=self.dec4(out2,adj)

        return out2
        
    
    def forward(self,x,adj):
        mu, logvar = self.encode(x,adj)     ## mu, log sigma 
        z = self.reparametrize(mu, logvar) ## z = mu + eps*sigma 
        #z=z.reshape(self.batch_size,2000,self.out_channels2)
        ##z=z.reshape(self.batch_size,2000,self.out_channels2)
        z2=self.decode(z,adj)
        #print(z2.shape)
        ##z3=z2[:,:,[0,1,2]] ## [0,1,2]
        #z3=F.sigmoid(z3)
        ##z4=z2[:,:,3]#3
        #z4=nn.Softmax(z4)
        ##z4=z4.unsqueeze_(2)
        #z4=self.lin3(z4)
        #z4=self.lin32(z4)
        return z2[:,[0,1]],z2[:,2], mu, logvar  ## Dropout z --> Adj = mult(z, z.transp) , mu , log var 
