import torch
import torch.nn as nn

mse_loss_fct = nn.MSELoss(reduction='mean')
bce_loss_fct = nn.BCEWithLogitsLoss(reduction='mean')
ce_loss_fct  = nn.CrossEntropyLoss(reduction='mean')

def pointwise_mse(logits, labels):
    scores = torch.sigmoid(logits)
    return mse_loss_fct(scores, labels)

def pointwise_bce(logits, labels):
    return bce_loss_fct(logits, labels)

def pairwise_hinge(logits, labels, group_size, tao=0.1):
    # Only support fp16 for now
    grouped_logits = logits.view(-1, group_size)
    grouped_labels = labels.view(-1, group_size)
    
    pos_logits = grouped_logits[:, 0:1] # batch_size x 1
    neg_logits = grouped_logits[:, 1:] # batch_size x (group_size - 1)
    loss = torch.clamp(tao + neg_logits - pos_logits, min=0).view(-1) # batch_size * group_size
    
    pos_label = grouped_labels[:, 0:1]
    neg_labels = grouped_labels[:, 1:]
    weight = (pos_label - neg_labels).view(-1) # batch_size * group_size
    weighted_loss = loss * weight
    non_zero_count = (weighted_loss != 0).sum()
    return (weighted_loss.sum() / non_zero_count) if non_zero_count > 0 else weighted_loss.sum()

def listwise_ce(logits, labels, group_size):
    grouped_logits = logits.view(-1, group_size)
    target = torch.zeros(grouped_logits.shape[0], dtype=torch.long).to(grouped_logits.device)
    # hard label
    loss = ce_loss_fct(grouped_logits, target)
    
    grouped_labels = labels.view(-1, group_size)
    grouped_labels = torch.softmax(grouped_labels.detach(), dim=-1).to(grouped_logits.device)
    # soft label
    loss += - torch.mean(torch.sum(torch.log_softmax(grouped_logits, dim=-1) * grouped_labels, dim=-1))
    return loss