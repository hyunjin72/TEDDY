import math
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required

def acosh(x):
    c0 = torch.log(x)
    c1 = torch.log1p(torch.sqrt(x * x - 1) / x)
    return c0 + c1

class CustomAdam(Optimizer):
    r""" Implements Adam with L0 regularization
    
    A General Family of Proximal Methods for Stochastic Preconditioned Gradient Descent

    This family contains the Adam-type curvature estimate and
    L0 (non-convex, non-smooth) regularizer

    For this optimizer, the update rule is somewhat adaptive hard-thresholding
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, l1_penalty=0.0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        l1_penalty=l1_penalty)
        super(CustomAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CustomAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def step(self, closure=None):
        """ Performs a single optimization step.
        
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
            
                state['step'] += 1 
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Perform weight decay
                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1
                p.data.addcdiv_(step_size, exp_avg, denom)
                p.step_vector = step_size * exp_avg / denom
                #step_vector = step_size * exp_avg / denom
                #p.step_vector = step_vector
        return loss
    
    def detaching(self):
        """ Performs a single optimization step.
        
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) != 0:
                    state['exp_avg'] = state['exp_avg'].detach()
                    state['exp_avg_sq'] = state['exp_avg_sq'].detach()

        return