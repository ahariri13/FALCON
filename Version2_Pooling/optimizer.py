import torch
import torch.nn.modules.loss
import torch.nn.functional as F
#import neuralnet_pytorch
import torch_geometric

def MAPELoss(output, target):
  return torch.mean(torch.abs((target - output) / target))   


def maeWeight(output, target):
  return torch.mean(torch.abs(target - output)^3)






def loss_function(r1,labels, batch,mu,sig):
    #cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    #cost1=cutLoss(r1,labels)
    loss2 = torch.nn.SmoothL1Loss()
    mae=torch.nn.L1Loss()
    #mape =torch.abs(torch.mean( labels[:,2]- r2) / labels[:,2])* 100
    lossMSE=torch.nn.MSELoss()

    #labels2=labels.reshape(batch,2000,3)
    #where=labels[:,0].nonzero().squeeze_(1)
    #r1=r1.reshape(batch*2000,2)
    #r2=r2.reshape(batch*2000)
    
    """
    labels2=labels[labels.sum(dim=1) != 0].clone()
    
    r1Full=r1.reshape(batch*2000,2)
    r1Full2=r1Full[r1Full.sum(dim=1) != 0].clone()
    
    r2Full=r2.reshape(batch*2000,1)
    r2Full2=r2Full[r2Full.sum(dim=1) != 0].clone()
    """

    #r1B=torch_geometric.utils.to_dense_batch(r1, batch=batch)
  
    #labelsB=torch_geometric.utils.to_dense_batch(labels, batch=batch)

    #cost1 =  neuralnet_pytorch.metrics.emd_loss(r1[0], labels[0], reduce='mean', sinkhorn=True)#lossMSE(r1, labels)
    #cost1 =  neuralnet_pytorch.metrics.chamfer_loss(r1B[0], labelsB[0],reduce='sum', c_code=True)
    #cost1=neuralnet_pytorch.metrics.emd_loss(r1, labels, reduce='mean', sinkhorn=True)

    cost1=lossMSE(r1,labels)
    KLD = -0.5 *(torch.mean(torch.sum(1 + sig - mu.pow(2) - sig.exp(), 1)))

    #KLD = -0.5 *(torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)))
    return cost1  + KLD 
