# https://github.com/jiawei-ren/BalancedMSE
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.distributions import MultivariateNormal as MVN

def bmc_loss(pred, target, noise_var):
    """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, 1].
      target: A float tensor of size [batch, 1].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    logits = - (pred - target.T).pow(2) / (2 * noise_var)   # logit size: [batch, batch]
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]))     # contrastive-like loss
    loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable 

    return loss

def bmse_loss(inputs, targets, noise_sigma=8.):
    inputs = inputs.reshape(inputs.shape[0],1)
    targets = targets.reshape(targets.shape[0],1)
    return bmc_loss(inputs, targets, noise_sigma ** 2)


class BMCLossMD(_Loss):
    """
    Multi-Dimension version BMC, compatible with 1-D BMC
    """

    def __init__(self, init_noise_sigma):
        super(BMCLossMD, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = bmc_loss_md(pred, target, noise_var)
        return loss


def bmc_loss_md(pred, target, noise_var):
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)

    """Compute the Multidimensional Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, d].
      target: A float tensor of size [batch, d].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    device = 'cpu'#pred.device
    pred = pred.to(device)
    target = target.to(device)
    noise_var = noise_var.to(device)
    I = torch.eye(pred.shape[-1]).to(device)
    logits = MVN(pred.unsqueeze(1), noise_var*I).log_prob(target.unsqueeze(0)).to(device)
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).to(device))
    loss = loss * (2 * noise_var).to(device).detach()
    return loss
