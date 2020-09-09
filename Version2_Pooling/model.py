import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
#from layers import GraphConvolution
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
import torch_geometric
import torch_geometric.nn as tnn

from torch_geometric.nn import EdgeConv, NNConv, GraphConv
from torch_geometric.nn.pool.edge_pool import EdgePooling

from torch_sparse import spspmm
from torch_geometric.nn import TopKPooling, GCNConv,GatedGraphConv, SAGPooling
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch_geometric.utils.repeat import repeat


class GraphPooling(nn.Module):
    def __init__(self,in_channels, out_channels1, out_channels2,out_channels3,out_channels4, out_channels5, dropout,batch_size):
        super(GraphPooling, self).__init__()   

        self.batch_size=batch_size
        self.out_channels2=out_channels2

        """
        Encoding
        """
        ### Encoding
        self.sage1=tnn.SAGEConv(in_channels,out_channels1,normalize=False)
        self.sage2=tnn.SAGEConv(out_channels1,out_channels2,normalize=False)
        self.sage3=tnn.SAGEConv(out_channels2,out_channels3,normalize=False)
        self.sage4=tnn.SAGEConv(out_channels3,out_channels4,normalize=False)
        self.tr1=nn.Linear(out_channels4,out_channels5)
        self.tr2=nn.Linear(out_channels5,2048)


        ## Batch Normalization
        self.bano1 = nn.BatchNorm1d(num_features=out_channels1)
        self.bano2 = nn.BatchNorm1d(num_features=out_channels2)
        self.bano3 = nn.BatchNorm1d(num_features=out_channels3)
        self.bano4 = nn.BatchNorm1d(num_features=out_channels4)
        self.bano5 = nn.BatchNorm1d(num_features=out_channels5)


        """
        Decoding
        """
        self.dec0= nn.Linear(2048,out_channels5)
        self.dec1= nn.Linear(out_channels5,out_channels4)
        self.dec2=tnn.SAGEConv(out_channels4,out_channels3,normalize=False)
        self.dec3=tnn.SAGEConv(out_channels3,out_channels2,normalize=False)
        self.dec4=tnn.SAGEConv(out_channels2,out_channels1,normalize=False)
        self.dec5=tnn.SAGEConv(out_channels1,in_channels,normalize=False)

     
        self.edge1=EdgePooling(out_channels1, edge_score_method=None, dropout=0., add_to_edge_score=0.5)
        self.edge2=EdgePooling(out_channels2, edge_score_method=None, dropout=0., add_to_edge_score=0.5)
        self.edge3=EdgePooling(out_channels3, edge_score_method=None, dropout=0., add_to_edge_score=0.5)


    def encode(self,x,adj,lengs):  

        
        hidden1=self.sage1(x,adj)
        hidden1=self.bano1(hidden1)
        hidden1=F.relu(hidden1)

        hidden1, edge_index, batch,info1=self.edge1(hidden1,adj,lengs)
        

        ### 2
        hidden1=self.sage2(hidden1,edge_index)
        hidden1=self.bano2(hidden1)
        hidden1=F.relu(hidden1) 

        hidden1, edge_index, batch,info2=self.edge2(hidden1,edge_index,batch)

              
        ### 3
        hidden1=self.sage3(hidden1,edge_index)
        hidden1=self.bano3(hidden1)
        hidden1=F.relu(hidden1)

        
        hidden1, edge_index, batch,info3=self.edge3(hidden1,edge_index,batch)
        
        ### 4 
        hidden1=self.sage4(hidden1,edge_index)
        hidden1=self.bano4(hidden1)
        hidden1=F.relu(hidden1) 
        
        ### 5
        hidden1=self.tr1(hidden1)
        hidden1=self.bano5(hidden1)
        hidden1=F.relu(hidden1) 
                
 
        return self.tr2(hidden1),self.tr2(hidden1),info1,info2, info3,edge_index#self.tr3(pooled[0]),self.tr3(pooled[0]),info

    def reparametrize(self, mu, logvar):
      if self.training:
          return mu + torch.randn_like(logvar) * torch.exp(logvar)
      else:
          return mu  
    
    def decode(self,z,adj,odd,info1,info2,info3,edgeInd):

        out1=self.dec0(z)
        out1=self.bano5(out1)
        out1=F.relu(out1)
        
        out1=self.dec1(out1)
        out1=self.bano4(out1)
        out1=F.relu(out1)

        out1=self.dec2(out1,edgeInd)
        out1=self.bano3(out1)
        out1=F.relu(out1)
        
        out1,edge_index,batch=self.edge3.unpool(out1,info3)

        out1=self.dec3(out1,edge_index)
        out1=self.bano2(out1)
        out1=F.relu(out1)  

        out1,edge_index,batch=self.edge2.unpool(out1,info2)


        out2=self.dec4(out1,edge_index)
        out2=self.bano1(out2)
        out2=F.relu(out2) 

        out2,egde_index,batch=self.edge1.unpool(out2,info1)

        ###
        out2=self.dec5(out2,adj)
        return out2
        
    
    def forward(self,x,adj,lengs,size):
        mu, logvar,info1,info2,info3,edgeInd = self.encode(x,adj,lengs)     ## mu, log sigma 
        z = self.reparametrize(mu, logvar) ## z = mu + eps*sigma 
        z2=self.decode(z,adj,size,info1,info2,info3,edgeInd)
        #z2=z2.reshape(self.batch_size,2000,3)
        return z2, mu, logvar 
