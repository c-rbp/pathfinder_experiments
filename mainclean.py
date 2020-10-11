#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:04:57 2019

"""

import os
import time
import torch
from torchvision.transforms import Compose as transcompose
import torch.nn.parallel
import torch.optim
import numpy as np

from utils.dataset import DataSetSeg
from models.hgrucleanSEG import hConvGRU, hConvGRUallSUP, hConvGRUtrunc
from models.convlstm import ConvLSTM
from models.FFnet import FFConvNet

from utils.transforms import GroupScale, Augmentation, Stack, ToTorchFormatTensor
from utils.misc_functions import AverageMeter, FocalLoss, acc_scores, save_checkpoint
from statistics import mean
from utils.opts import parser
import matplotlib
import imageio
from torch._six import inf
matplotlib.use('Agg')


torch.backends.cudnn.benchmark = True

global best_prec1
best_prec1 = 0
args = parser.parse_args()
transform_list = transcompose([GroupScale((150, 150)), Augmentation(), Stack(), ToTorchFormatTensor(div=True)])

pf_root = '/users/akarkada'
#pf_root = '/gpfs/data/tserre/data/lgovinda/'
print("Loading training dataset")
train_loader = torch.utils.data.DataLoader(DataSetSeg(pf_root, args.train_list, transform=transform_list),
                                           batch_size=args.batch_size, shuffle=True, num_workers=8,
                                           pin_memory=True, drop_last=True)

print("Loading validation dataset")
val_loader = torch.utils.data.DataLoader(DataSetSeg(pf_root, args.val_list, transform=transform_list),
                                         batch_size=args.batch_size, shuffle=False, num_workers=4,
                                         pin_memory=False, drop_last=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def validate(val_loader, model, criterion, logiters=None):
    batch_timev = AverageMeter()
    lossesv = AverageMeter()
    top1v = AverageMeter()
    precisionv = AverageMeter()
    recallv = AverageMeter()
    f1scorev = AverageMeter()

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, target) in enumerate(val_loader):
            target = target.cuda()
            target = (target > 0.2).squeeze().long()
            imgs = imgs.cuda()
            output, gt2, loss = model.forward(imgs, 0, 0, target, criterion)
            
            loss = loss.mean()
            prec1, preci, rec, f1s = acc_scores(target, output.data)
            
            lossesv.update(loss.data.item(), 1)
            top1v.update(prec1.item(), 1)
            precisionv.update(preci.item(), 1)
            recallv.update(rec.item(), 1)
            f1scorev.update(f1s.item(), 1)
            
            batch_timev.update(time.time() - end)
            end = time.time()

            if (i % args.print_freq == 0 or (i == len(val_loader) - 1)) and logiters is None:

                print_string = 'Test: [{0}/{1}]\t Time: {batch_time.avg:.3f}\t Loss: {loss.val:.8f} ({loss.avg: .8f})\t'\
                               'Bal_acc: {balacc:.8f} preci: {preci.val:.5f} ({preci.avg:.5f}) rec: {rec.val:.5f}'\
                               '({rec.avg:.5f}) f1: {f1s.val:.5f} ({f1s.avg:.5f})'\
                               .format(i, len(val_loader), batch_time=batch_timev, loss=lossesv, balacc=top1v.avg,
                                       preci=precisionv, rec=recallv, f1s=f1scorev)
                print(print_string)
                with open(results_folder + args.name + '.txt', 'a+') as log_file:
                    log_file.write(print_string + '\n')

            elif logiters is not None:
                if i > logiters:
                    break
    model.train()
    return top1v.avg, precisionv.avg, recallv.avg, f1scorev.avg, lossesv.avg


def save_npz(epoch, log_dict, results_folder, savename='train'):

    with open(results_folder + savename + '.npz', 'wb') as f:
        np.savez(f, **log_dict)


if __name__ == '__main__':
    
    results_folder = 'results/{0}/'.format(args.name)
    os.mkdir(results_folder)
    
    exp_logging = args.log
    jacobian_penalty = args.penalty

    timesteps = 6
    if args.model == 'hgru':
        print("Init model hgru ", args.algo, 'penalty: ', args.penalty, 'steps: ', timesteps)
        model = hConvGRU(timesteps=timesteps, filt_size=15, num_iter=15, exp_name=args.name, jacobian_penalty=jacobian_penalty,
                         grad_method=args.algo)
    elif args.model == 'clstm':
        print("Init model clstm ", args.algo, 'penalty: ', args.penalty, 'steps: ', timesteps)
        model = ConvLSTM(timesteps=timesteps, filt_size=15, num_iter=15, exp_name=args.name, jacobian_penalty=jacobian_penalty,
                         grad_method=args.algo)
    elif args.model == 'ff':
        print("Init model feedforw ", args.algo)
        model = FFConvNet(filt_size=15)
    else:
        print('Model not found')
    print(sum([p.numel() for p in model.parameters() if p.requires_grad]))
    if args.parallel is True:
        model = torch.nn.DataParallel(model).to(device)
        print("Loading parallel finished on GPU count:", torch.cuda.device_count())
    else:
        model = model.to(device)
        print("Loading finished")

    criterion = FocalLoss(gamma=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.7)
    lr_init = args.lr

    val_log_dict = {'loss': [], 'balacc': [], 'precision': [], 'recall': [], 'f1score': []}
    train_log_dict = {'loss': [], 'balacc': [], 'precision': [], 'recall': [], 'f1score': [], 'jvpen': [], 'scaled_loss': []}


    exp_loss = None
    scale = torch.Tensor([1.0]).to(device)
    for epoch in range(args.start_epoch, args.epochs):
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        precision = AverageMeter()
        recall = AverageMeter()
        f1score = AverageMeter()

        time_since_last = time.time()
        model.train()
        end = time.perf_counter()

        for i, (imgs, target) in enumerate(train_loader):
            data_time.update(time.perf_counter() - end)
            
            imgs = imgs.to(device)
            target = target.to(device)
            target = (target > 0.2).squeeze().long()

            output, jv_penalty, loss = model.forward(imgs, epoch, i, target, criterion)
            loss = loss.mean()
            losses.update(loss.data.item(), 1)
            jv_penalty = jv_penalty.mean()
            train_log_dict['jvpen'].append(jv_penalty.item())

                
            if jacobian_penalty:
                loss = loss + jv_penalty * 1e1
            
            prec1, preci, rec, f1s = acc_scores(target[:], output.data[:])
            
            top1.update(prec1.item(), 1)
            precision.update(preci.item(), 1)
            recall.update(rec.item(), 1)
            f1score.update(f1s.item(), 1)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            batch_time.update(time.perf_counter() - end)
            
            end = time.perf_counter()
            if exp_logging and i % 20 == 0:
                accv, precv, recv, f1sv, losv = validate(val_loader, model, criterion, logiters=3)
                print('val f', f1sv)
                val_log_dict['loss'].append(losv)
                val_log_dict['balacc'].append(accv)
                val_log_dict['precision'].append(precv)
                val_log_dict['recall'].append(recv)
                val_log_dict['f1score'].append(f1sv)

            if i % (args.print_freq) == 0:
                time_now = time.time()
                print_string = 'Epoch: [{0}][{1}/{2}]  lr: {lr:g}  Time: {batch_time.val:.3f} (itavg:{timeiteravg:.3f}) '\
                               '({batch_time.avg:.3f})  Data: {data_time.val:.3f} ({data_time.avg:.3f}) ' \
                               'Loss: {loss.val:.8f} ({lossprint:.8f}) ({loss.avg:.8f})  bal_acc: {top1.val:.5f} '\
                               '({top1.avg:.5f}) preci: {preci.val:.5f} ({preci.avg:.5f}) rec: {rec.val:.5f} '\
                               '({rec.avg:.5f})  f1: {f1s.val:.5f} ({f1s.avg:.5f}) jvpen: {jpena:.12f} {timeprint:.3f} losscale:{losscale:.5f}'\
                               .format(epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses,
                                       lossprint=mean(losses.history[-args.print_freq:]), lr=optimizer.param_groups[0]['lr'],
                                       top1=top1, timeiteravg=mean(batch_time.history[-args.print_freq:]),
                                       timeprint=time_now - time_since_last, preci=precision, rec=recall,
                                       f1s=f1score, jpena=jv_penalty.item(), losscale=scale.item())
                print(print_string)
                time_since_last = time_now
                with open(results_folder + args.name + '.txt', 'a+') as log_file:
                    log_file.write(print_string + '\n')
        #lr_scheduler.step()

        train_log_dict['loss'].extend(losses.history)
        train_log_dict['balacc'].extend(top1.history)
        train_log_dict['precision'].extend(precision.history)
        train_log_dict['recall'].extend(recall.history)
        train_log_dict['f1score'].extend(f1score.history)
        save_npz(epoch, train_log_dict, results_folder, 'train')
        save_npz(epoch, val_log_dict, results_folder, 'val')

        if (epoch + 1) % 1 == 0 or epoch == args.epochs - 1:
            _, _, _, f1va, _ = validate(val_loader, model, criterion)
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_prec1': f1va}, True, results_folder)
