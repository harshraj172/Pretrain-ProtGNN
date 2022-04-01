import torch
import torch.nn.functional as F
import GCL.losses as L

def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()

class InfoNCE(Loss):
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau
    
    def compute(self, anchor, sample, pos_mask, neg_mask, topk, *args, **kwargs):
        sim = _similarity(anchor, sample) / self.tau
        # Loss 
        exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        # Accuracy
        acc = accuracy(sim, topk)
        return -loss.mean(), acc[0]