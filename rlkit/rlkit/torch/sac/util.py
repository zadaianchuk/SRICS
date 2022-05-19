import math
import rlkit.torch.pytorch_util as ptu
import torch
from torch import nn as nn

def bug_schedule(epoch, log_alpha, alpha_optimizer: torch.optim.Adam):
    reset_epoch = 4
    if epoch == reset_epoch:
        alpha = 1 
        log_alpha = ptu.ones(1).fill_(math.log(alpha))
        log_alpha = nn.Parameter(log_alpha)
        alpha_optimizer = torch.optim.Adam([log_alpha], lr=alpha_optimizer.param_groups[0]["lr"])
        return log_alpha, alpha_optimizer
    else:
        return log_alpha, alpha_optimizer

def sqrt_schedule(epoch, alpha):
    return alpha/(epoch**(-1/2))

def exp_schedule(epoch, alpha):
    return alpha * (2**(-epoch))

schedules = {"bug": bug_schedule, "sqrt": sqrt_schedule, "exp": exp_schedule}

    
if __name__ == '__main__':
    
    alpha = 1 
    log_alpha = ptu.ones(1).fill_(math.log(alpha))
    log_alpha = nn.Parameter(log_alpha)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=0.001)
    log_alpha, alpha_optimizer = bug_schedule(4, log_alpha, alpha_optimizer)
    assert alpha_optimizer.param_groups[0]["lr"] == 0.001


