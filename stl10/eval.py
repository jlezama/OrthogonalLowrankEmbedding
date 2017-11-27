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
import numpy as np
import scipy.io



parser = argparse.ArgumentParser(description='PyTorch STL10 Example')
parser.add_argument('--channel', type=int, default=32, help='first conv channel (default: 32)')
parser.add_argument('--wd', type=float, default=0.00, help='weight decay')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=164, help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 1e-3)')
parser.add_argument('--lambda_', type=float, default=0.25, help='OLE Loss weight (default: 0.25)')
parser.add_argument('--gpu', default='0', help='index of gpus to use')
parser.add_argument('--ngpu', type=int, default=2, help='number of gpus to use')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=125,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=5,  help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='results/b09d0499-7b38-4cc5-8b82-87ec1052ab4a', help='folder where the checkpoint is saved')
parser.add_argument('--decreasing_lr', default='81,122', help='decreasing strategy')


args = parser.parse_args()

foldername =  args.logdir


args.logdir = os.path.join(os.path.dirname(__file__), foldername)
misc.logger.init(args.logdir, 'train_log_eval_%06i' % (np.random.randint(99999999)))
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
# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)

# data loader and model
train_loader, test_loader = dataset.get(batch_size=args.batch_size, num_workers=1)
model = model.stl10(n_channel=args.channel)


model = torch.nn.DataParallel(model, device_ids= range(args.ngpu))

# load model

model.module.load_state_dict(torch.load('%s/checkpoint-163.pth' % foldername))

if args.cuda:
    print('USING CUDA')
    model.cuda()

best_acc, old_file = 0, None
t_begin = time.time()
crit0 = nn.CrossEntropyLoss()
crit1 = OLELoss(lambda_=args.lambda_)
#try:
if 1:
    # ready to go
    model.eval()
    test_loss = 0
    correct = 0


    features = None
    labels = None

    for data, target in test_loader:
        indx_target = target.clone()
        if args.cuda:
            data, target = data.cuda(), target.cuda().long().squeeze()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        

        test_loss += crit1(output[1], target)


        pred = output[0].data.max(1)[1]  # get the index of the max log-probability
        correct += pred.cpu().eq(indx_target).sum()

        # save features
        featuresi = output[1].data.cpu().numpy()
        labelsi = target.data.cpu().numpy()
        
        if features is None:
            features = featuresi
            labels = labelsi
        else:
            features = np.concatenate((features, featuresi), 0)
            labels = np.concatenate((labels, labelsi), 0)
    
    features_test = features
    labels_test = labels

    print((features_test.shape, labels_test.shape))
    test_loss = test_loss.data[0] / len(test_loader) # average over number of mini-batch
    acc = 100. * correct / len(test_loader.dataset)

    print('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
                test_loss, correct, len(test_loader.dataset), acc))

    # save result to matlab .MAT file
    epoch = 163
    scipy.io.savemat('%s/%05i.mat' % (foldername, epoch), mdict={'features_test':features_test, 'labels_test':labels_test})





