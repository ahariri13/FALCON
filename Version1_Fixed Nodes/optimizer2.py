import torch
import torch.nn.modules.loss
import torch.nn.functional as F


def MAPELoss(output, target):
  return torch.mean(torch.abs((target - output) / target))   


def maeWeight(output, target):
  return torch.mean(torch.abs(target - output)^3)




def loss_function(r1,r2,labels, mu, logvar, n_nodes):
    #cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    #cost1=cutLoss(r1,labels)
    loss2 = torch.nn.SmoothL1Loss()
    mae=torch.nn.L1Loss()
    mape =torch.abs(torch.mean( labels[:,2]- r2) / labels[:,2])* 100
    lossMSE=torch.nn.MSELoss()
    cost1 =  lossMSE(r1, labels[:,[0,1]])+ lossMSE(labels[:,2],r2)#loss2(r2,labels[:,2])
   
    KLD = -0.5 *(torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)))
    return cost1  + KLD 
