"""
OLE training script
"""


from __future__ import print_function

import argparse
import os, sys
import time
from utee import misc

import torch
import torch.nn as  nn
import torch.optim as optim
from torch.autograd import Variable

import dataset
import model
from IPython import embed


# OLE Loss
from OLE import *
import uuid
import sys
from datetime import datetime



parser = argparse.ArgumentParser(description='PyTorch STL10 with OLE')
parser.add_argument('--channel', type=int, default=32, help='first conv channel (default: 32)')
parser.add_argument('--wd', type=float, default=1e-3, help='weight decay')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=164, help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 1e-3)')
parser.add_argument('--lambda_', type=float, default=0.25, help='OLE loss weight \lambda (default: 0.25)')
parser.add_argument('--gpu', default='0', help='index of gpus to use')
parser.add_argument('--ngpu', type=int, default=2, help='number of gpus to use')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--num_samples', type=int, default=500, help='number of training samples per class to use')
parser.add_argument('--data_augment', type=int, default=1, help='use data augmentation, 1: yes (default), 0: no')
parser.add_argument('--validation', type=int, default=0, help='run validation on 10%% of training set 0: no (default), 1: yes')
parser.add_argument('--log_interval', type=int, default=125,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=5,  help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument('--decreasing_lr', default='81,122', help='decreasing strategy')
args = parser.parse_args()

foldername =  'results/wd_%s_batch_%s_channel_%s_samples_%s/' % (str(args.wd), str(args.batch_size), str(args.channel), str(args.num_samples)) + str(uuid.uuid4())


args.logdir = os.path.join(os.path.dirname(__file__), foldername)
misc.logger.init(args.logdir, 'train_log')
print = misc.logger.info

# select gpu
args.gpu = misc.auto_select_gpu(utility_bound=0, num_gpu=args.ngpu, selected_gpus=args.gpu)
args.ngpu = len(args.gpu)

# logger
misc.ensure_dir(args.logdir)
print("=================FLAGS==================")
for k, v in args.__dict__.items():
    print('{}: {}'.format(k, v))
print("========================================")

# seed
args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# data loader and model
train_loader, test_loader = dataset.get(batch_size=args.batch_size, num_workers=1, num_samples=args.num_samples, data_augment=args.data_augment, validation=args.validation)

if args.validation or (args.num_samples!=500):
    Ntrain = len(train_loader.sampler.indices)
else:
    Ntrain = len(train_loader.dataset)

if args.validation:
    Ntest = len(test_loader.sampler.indices)
else:
    Ntest = len(test_loader.dataset)


model = model.stl10(n_channel=args.channel)
model = torch.nn.DataParallel(model, device_ids= range(args.ngpu))
if args.cuda:
    print('USING CUDA')
    model.cuda()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
print('decreasing_lr: ' + str(decreasing_lr))
best_acc, old_file = 0, None
t_begin = time.time()
crit0 = nn.CrossEntropyLoss()
crit1 = OLELoss(lambda_=args.lambda_)



# ready to go
for epoch in range(args.epochs):
    model.train()
    if epoch in decreasing_lr:
        optimizer.param_groups[0]['lr'] *= 0.1
    for batch_idx, (data, target) in enumerate(train_loader):
        indx_target = target.clone()

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()

        output = model(data)

        loss = crit0(output[0], target) 
        OLE_loss = crit1(output[1], target)
        
        if args.lambda_ >0:
            loss += OLE_loss

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            pred = output[0].data.max(1)[1]  # get the index of the max log-probability
            correct = pred.cpu().eq(indx_target).sum()
            acc = correct * 1.0 / len(data)
            print('Train Epoch: {} [{}/{}] Loss: {:.6f} OLE Loss: {:.6f} Acc: {:.4f} lr: {:.2e}'.format(
                epoch, batch_idx * len(data), Ntrain,
                loss.data[0], OLE_loss.data[0], acc, optimizer.param_groups[0]['lr']))

    elapse_time = time.time() - t_begin
    speed_epoch = elapse_time / (epoch + 1)
    speed_batch = speed_epoch / len(train_loader)
    eta = speed_epoch * args.epochs - elapse_time
    print("Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
        elapse_time, speed_epoch, speed_batch, eta))
    misc.model_snapshot(model, os.path.join(args.logdir, 'latest.pth'))

    if epoch % args.test_interval == 0:
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            indx_target = target.clone()
            if args.cuda:
                data, target = data.cuda(), target.cuda().long().squeeze()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)

            test_loss += crit0(output[0], target)
            test_OLE_loss =  crit1(output[1], target)
            
            if args.lambda_ >0:
                 test_loss += test_OLE_loss

            # test_loss += F.cross_entropy(output, target).data[0] 
            pred = output[0].data.max(1)[1]  # get the index of the max log-probability
            correct += pred.cpu().eq(indx_target).sum()

        test_loss = test_loss.data[0] / len(test_loader) # average over number of mini-batch
        test_OLE_loss = test_OLE_loss.data[0] # already averaged over minibatch

        test_acc = 100. * correct / float(Ntest)

        print('\tTest set {}/{}: Average loss: {:.4f}, OLE loss: {:.4f}  Accuracy: {}/{} ({:.2f}%)'.format( epoch, args.epochs, test_loss, test_OLE_loss, correct, Ntest, test_acc))



new_file = os.path.join(args.logdir, 'checkpoint-{}.pth'.format(epoch))
misc.model_snapshot(model, new_file, verbose=True)
print("Total Elapse: {:.2f}, Final Test Result: {:.3f}%".format(time.time()-t_begin, test_acc))


