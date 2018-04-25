'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017

Modifications for use with Orthogonal Lowrank Embedding by Jose Lezama, 2017
'''
# from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


# OLE Loss
from OLE import *
from backup import *
import uuid
import sys
from datetime import datetime
import numpy as np
import scipy.io
from torch.utils.data.sampler import SubsetRandomSampler



model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=164, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=64, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[81, 122],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet)')
parser.add_argument('--depth', type=int, default=110, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# Device options
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# OLE Loss
parser.add_argument('--lambda_', type=float, default=0.25, help='OLE loss is multiplied by lambda_.')
parser.add_argument( '--no_augment', dest='no_augment', action='store_true',
                    help='do not perform data augmentation')
parser.add_argument( '--validation', dest='validation', action='store_true',
                     help='do validation (trains on 90%% of training set, evaluates on remaining 10%%)')
parser.add_argument( '--no_print', dest='no_print', action='store_true',
                    help='do not print to file')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy
use_OLE = False

def main():

    #####################
    # START SETUP LOGGING

    foldername =  str(uuid.uuid4())

    isplus = '+' if  not args.no_augment else ''
    
    savefolder = 'results/%s/%s%s/%s' % (args.arch, args.dataset, isplus, foldername)
    os.system('mkdir -p %s' % savefolder)
    args.checkpoint = savefolder
    
    time.sleep(5) # wait for directory creation
    print 'folder is ', foldername

    # use sys.stdout to log to file
    orig_stdout = sys.stdout
    logfilename = '%s/log.txt' % (savefolder)
    logfile = file(logfilename, 'w')

    if not args.no_print:
        print 'Printing to file %s' % logfilename
        sys.stdout = logfile
    else:
        print 'Printing to stdout'

    backupfname = '%s/code_snapshot_%s.zip' % (savefolder, str(datetime.now()))
    backupfname = backupfname.replace(' ','_')
    backup_code(backupfname, '.', ['.py'], ['result', 'log',])

    print args
    # END SETUP LOGGING
    ###################



    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)



    # Data
    print('==> Preparing dataset %s' % args.dataset)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.no_augment:
        print 'NO DATA AUGMENTATION'
        transform_train = transform_test
    else:
        print 'USE DATA AUGMENTATION'
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100


    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)

    if args.validation:
        # select random subset for validation
        N = len(trainset)
        train_size = int(N*.9) # use 90 % of training set for training
        valid_size = N-train_size
    
        print 'number of training examples is %i/%i' % (train_size,N)
        indices = torch.randperm(N)
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:]

        assert(set(train_indices).isdisjoint(set(valid_indices)))
        
        trainloader = data.DataLoader(trainset, batch_size=args.train_batch, sampler=SubsetRandomSampler(train_indices), num_workers=args.workers)
        testloader = data.DataLoader(trainset, batch_size=args.test_batch, sampler=SubsetRandomSampler(valid_indices), num_workers=args.workers)

    else:
        trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
        testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)



    
    # Model   
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )        
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # use OLE loss?
    print  'lambda_ =', args.lambda_
    lambda_ = args.lambda_

    global use_OLE

    print args
    
    if lambda_>0:
        use_OLE = True
    else:
        use_OLE = False


    if use_OLE:    
        criterion = [nn.CrossEntropyLoss()] + [OLELoss(lambda_=args.lambda_)]
    else:
        criterion = [nn.CrossEntropyLoss()] 

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.join(os.path.dirname(args.resume), 'fine_tune')

        if not os.path.isdir(args.checkpoint):
            mkdir_p(args.checkpoint)


        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = 0 # checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        # logger = Logger(os.path.join(args.checkpoint, 'log_finetune.txt'), title=title)
        # logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
    else:
        pass
        # logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        # logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f, lambda_: %f' % (epoch + 1, args.epochs, state['lr'], args.lambda_))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)

        # append logger file
        # logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    # logger.close()
    # logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

    #############
    # END LOGGING
    sys.stdout = orig_stdout
    logfile.close()

    print '---'
    print 'saved results to ', savefolder

    print('Done!')
    
    # LOGGING ENDED
    ###############
    

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()


    if len(criterion)==2:
        lambda_ = criterion[1].lambda_
    else:
        lambda_ = 0.
        
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    ole = AverageMeter()
    
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)

        # criterion is a list composed of crossentropy loss and OLE loss.
        losses_list = [-1,-1]
        # output_Var contains scores in the first element and features in the second element
        loss = 0
        for cix, crit in enumerate(criterion):
            losses_list [cix] = crit(outputs[cix], targets)
            loss += losses_list[cix]

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs[0].data, targets.data, topk=(1, 5))
        losses.update(losses_list[0].data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        if use_OLE:
            ole.update(losses_list[1].data[0], inputs.size(0))
        else:
            ole.update(-1, 1)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '[{epoch: d}] ({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f} | OLE: {ole: .4f} (lambda_ = {lambda_: .2f})'.format(
                    epoch=epoch,
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    ole=ole.avg,
                    lambda_ = lambda_,
                    )
        bar.next()
    if not args.no_print:
        print 'Train (%i): %s (lambda_ = %f)' % (epoch, bar.suffix, lambda_)
    bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    ole = AverageMeter()
    
    # switch to evaluate mode
    model.eval()

    if len(criterion)==2:
        lambda_ = criterion[1].lambda_
    else:
        lambda_ = 0
        
    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        # criterion is a list composed of crossentropy loss and OLE loss.
        losses_list = [-1,-1]
        # output_Var contains scores in the first element and features in the second element
        loss = 0
        for cix, crit in enumerate(criterion):
            losses_list [cix] = crit(outputs[cix], targets)
            loss += losses_list[cix]

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs[0].data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        if use_OLE:
            ole.update(losses_list[1].data[0], inputs.size(0))
        else:
            ole.update(-1, 1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '[{epoch: d}] ({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f} | OLE: {ole: .4f} (lambda_ = {lambda_: .2f})'.format(
                    epoch=epoch,
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    ole=ole.avg,
                    lambda_ = lambda_,
                    )
        bar.next()
    if not args.no_print:
        print 'Test (%i): %s (lambda_ = %f)' % (epoch, bar.suffix, lambda_)
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
