import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
from torch._six import inf


def metric_scores(target, pred):
    correct = pred.eq(target)
    tp = correct[target == 1].sum().float()
    tn = correct[target == 0].sum().float()

    P = target.sum()
    N = (target == 0).sum()
    tpfp = pred.sum()
    if tpfp == 0:
        tpfp = 1e-6
    recall = tp / P
    precision = tp / tpfp
    bacc = (tn / N + recall) / 2
    f1s = (2 * tp) / (P + tpfp)
    return bacc, precision, recall, f1s


def acc_scores(target, prediction):
    target = target.byte()
    _, pred = prediction.topk(1, 1, True, True)
    balacc, precision, recall, f1s = metric_scores(target, pred.squeeze().byte())
    return balacc * 100, precision, recall, f1s


def clip_grad_norm_(parameters, max_norm, i, norm_type=2, do=False):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        if not do:
            print('grad norm', total_norm, i)
        else:
            for p in parameters:
                p.grad.data.mul_(clip_coef)
    return total_norm


def save_checkpoint(state, is_best, results_folder):
    save_folder = results_folder + 'saved_models/'
    try:
        os.mkdir(save_folder)
    except:
        pass
    filename = save_folder + 'model_fscore_{0:04d}_epoch_{1:02d}_checkpoint.pth.tar'.format(
        int(state['best_prec1'] * 1e4), state['epoch'])
    torch.save(state, filename)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if alpha is not None:
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.history = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.history.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("biasqq" not in n) and (p.grad is not None):
            
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
            if p.grad.abs().mean() == 0:
                layers.append(n + "ZERO")
            elif p.grad.abs().mean() < 0.00001:
                layers.append(n + "SMALL")
            else:
                layers.append(n)
    
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.2) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
#    plt.draw()
#    plt.pause(0.001)
#    plt.tight_layout()
#    plt.savefig('grads.png')
    return 0