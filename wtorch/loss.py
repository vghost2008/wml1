import torch
import torch.nn.functional as F

def focal_loss_for_heat_map(labels,logits,pos_threshold=0.99,alpha=2,beta=4,sum=True):
    '''
    focal loss for heat map, for example CenterNet2's heat map loss
    '''
    logits = logits.to(torch.float32)
    zeros = torch.zeros_like(labels)
    ones = torch.ones_like(labels)
    num_pos = torch.sum(torch.where(torch.greater_equal(labels, pos_threshold), ones, zeros))

    probs = F.sigmoid(logits)
    pos_weight = torch.where(torch.greater_equal(labels, pos_threshold), ones - probs, zeros)
    neg_weight = torch.where(torch.less(labels, pos_threshold), probs, zeros)
    '''
    用于保证数值稳定性，log(sigmoid(x)) = log(1/(1+e^-x) = -log(1+e^-x) = x-x-log(1+e^-x) = x-log(e^x +1)
    pos_loss = tf.where(tf.less(logits,0),logits-tf.log(tf.exp(logits)+1),tf.log(probs))
    '''
    pure_pos_loss = -torch.minimum(logits,logits.new_tensor(0,dtype=logits.dtype))+torch.log(1+torch.exp(-torch.abs(logits)))
    pos_loss = pure_pos_loss*torch.pow(pos_weight, alpha)
    if sum:
        pos_loss = torch.sum(pos_loss)
    '''
    用于保证数值稳定性
    '''
    pure_neg_loss = F.relu(logits)+torch.log(1+torch.exp(-torch.abs(logits)))
    neg_loss = torch.pow((1 - labels), beta) * torch.pow(neg_weight, alpha) * pure_neg_loss
    if sum:
        neg_loss = torch.sum(neg_loss)
    loss = (pos_loss + neg_loss) / (num_pos + 1e-4)
    return loss