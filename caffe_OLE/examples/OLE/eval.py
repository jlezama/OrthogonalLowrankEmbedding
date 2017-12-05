""" Python script for testing low-rank networks with caffe 
must run this in corsair before running:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jose/Documents/code/anaconda2/lib/
export PYTHONPATH=/home/jose/Documents/code/caffe3/python/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jose/Documents/code/anaconda2/pkgs/jpeg-9b-0/lib/


in mbp:
export LD_LIBRARY_PATH=/usr/local/lib/:~/Documents/anii/code/anaconda2/lib/
export ANACONDA_HOME=~/Documents/anii/code/anaconda2
export DYLD_FALLBACK_LIBRARY_PATH=$ANACONDA_HOME/lib:/usr/local/lib:/usr/lib
export PYTHONPATH=/Users/jose/Documents/duke/code/caffe/python/
export PYTHONHOME=/Users/jose/Documents/anii/code/anaconda2/

setenv('ANACONDA_HOME', '~/Documents/anii/code/anaconda2')
setenv('DYLD_FALLBACK_LIBRARY_PATH', '$ANACONDA_HOME/lib:/usr/local/lib:/usr/lib')
setenv('PYTHONPATH', '/Users/jose/Documents/duke/code/caffe/python/')
setenv('PYTHONHOME', '/Users/jose/Documents/anii/code/anaconda2/')
setenv('LD_LIBRARY_PATH', '/usr/local/lib/:~/Documents/anii/code/anaconda2/lib/')

in medusa:
CAFFE_DIR = '/home/jlezama/code/caffe'
export PYTHONPATH=/home/jlezama/code/caffe/python/

"""

import glob
import sys
import os
import optparse
from datetime import datetime
from collections import OrderedDict
import time
from PIL import Image
import numpy as np
import scipy.io

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import sample_points
from collections import OrderedDict
################################################################################
# START IMPORT CAFFE

# set some useful environmental variables according to host computer
computer_name =  os.uname()[1].split('.')[0]
print computer_name

if computer_name == 'joses-mbp-2' or computer_name == 'laptop-iie-33':
  CAFFE_DIR = '/Users/jose/Documents/duke/code/caffe/'
  CS231nDIR = '/Users/jose/Documents/anii/code/20170116_CS321n/sandbox'
  SOLVER_MODE = 'CPU'
  os.environ['DYLD_FALLBACK_LIBRARY_PATH']= '/Users/jose/Documents/anii/code/anaconda2/lib:/usr/local/lib:/usr/lib'

elif computer_name == 'medusa13':
  CAFFE_DIR = '/home/jlezama/code/LargeMargin_Softmax_Loss'
  CS231nDIR = '/home/jlezama/code/caffe/examples/20170306_deepLRT/datasets/'
  SOLVER_MODE = 'GPU'

elif computer_name == 'corsair':
  CAFFE_DIR = '/home/jose/Documents/code/LargeMargin_Softmax_Loss/'
  CS231nDIR = '/home/jose/mbp/Documents/anii/code/20170116_CS321n/sandbox'
  SOLVER_MODE = 'GPU'
  os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH'] + ':/home/jose/Documents/code/anaconda2/lib/:/home/jose/Documents/code/anaconda2/pkgs/jpeg-9b-0/lib/'
  os.environ['PYTHONPATH'] = '/home/jose/Documents/code/LargeMargin_Softmax_Loss/python/'

else:
  raise ValueError('Unknown machine')
  CAFFE_DIR = '/home/jlezama/code/caffe'
  CS231nDIR = '/home/jlezama/code/caffe/examples/20170306_deepLRT/datasets/'
  SOLVER_MODE = 'GPU'  
  print 'Unrecognized computer, expect path problems'


sys.path.append('%s/python' % CAFFE_DIR)

import caffe


# END IMPORT CAFFE
################################################################################

from auxfunctions import *
from backup import *
from network_functions import *

import uuid


"""This code should replicate, using Caffe, the results obtained with the Numpy
only implementation in the Python notebook using CIFAR data

"""

################################################################################
# GET ARGUMENTS
# GPU ID

parser = optparse.OptionParser()
parser.add_option('-g', '--gpu_id', action='store', dest='gpu_id', help='ID of GPU device', default=0, type=int)
parser.add_option('-r', '--run_mode', action='store', dest='run_mode', help='0 for fixed parameters, 1 for random (default)', default=1, type=int)
parser.add_option('-d', '--dataset', action='store', dest='dataset', help='dataset to be used (cifar10*, cifar100', default="cifar10", type=str)
parser.add_option('-p', '--print_to_file', action='store', dest='print_to_file', help='1 to save log to file, 0 to print to stdout', default=1, type=int)
parser.add_option('-m', '--model', action='store', dest='model', help='network model to be used (custom*, allcnn)', default="custom", type=str)
parser.add_option('-w', '--weights', action='store', dest='weights', help='pretrained weights', default=None, type=str)
parser.add_option('-f', '--folder', action='store', dest='folder', help='folder to resume', default=None, type=str)



options, args = parser.parse_args()
# gpu_id = int(sys.argv[1]) if len(sys.argv)>1 else 0;
# run mode: 0: fixed parameters, 1: random (default)
# run_mode = int(sys.argv[2]) if len(sys.argv)>2 else 1; 
# dataset, options: 'cifar', 'wiki_images', 'wiki_text'
# dataset = int(sys.argv[3]) if len(sys.argv)>3 else 'cifar'; 

################################################################################
## INITIALIZE GPU
if SOLVER_MODE == 'CPU':
  caffe.set_mode_cpu() #for cpu mode do 'caffe.set_mode_cpu()'
  caffe_gpu_command_str = ''
elif SOLVER_MODE == 'GPU':
  caffe.set_mode_gpu() #for cpu mode do 'caffe.set_mode_cpu()'
  # get GPU id from sys.argv
  caffe_gpu_command_str = '-gpu %i' % options.gpu_id
  caffe.set_device(options.gpu_id)


else:
  raise ValueError('invalid solver mode')



################################################################################
# CREATE DIRECTORY, ETC
ROOT_DIR = '%s/myexamples/cifar10_2/' % CAFFE_DIR

RESULTS_DIR = '%s/results_comparison_%s' % (ROOT_DIR, options.dataset)

RESULTS_DIR = '/data/medusa13_backup/LargeMargin_Softmax_Loss/examples/20170321_deepLRT2/results_cifar10/';

os.system("mkdir -p %s" % RESULTS_DIR)


foldername = options.folder #  '8115b2e4-f43b-4da8-b01a-82c9156d572b'; # f35df507-a6ad-4bcc-a5db-556ebf2fd68c';

results_folder = '%s/%s' % (RESULTS_DIR, foldername)
# os.system('mkdir -p %s' % results_folder)

print 'folder is ', foldername

# use sys.stdout to log to file
orig_stdout = sys.stdout

PRINT_TO_FILE = options.print_to_file


################################################################################
# MAIN

if PRINT_TO_FILE:
  logfile = file('%s/log_eval.txt' % (results_folder), 'w')
  sys.stdout = logfile

print '........'
print 'STARTING'
print '........'
print
print '........'
print 'TAKING CODE SNAPSHOT'
print '........'
print
print 'folder name is %s' % foldername
print 'executed %s' % str(datetime.now())
print

# backupfname = '%s/code_snapshot_%s.zip' % (results_folder, str(datetime.now()))
# backupfname = backupfname.replace(' ','_')
# backup_code(backupfname, '.', ['.py', '.prototxt'], ['result', 'log',])


# backupfname = '%s/caffe_pythoncode_snapshot_%s.zip' % (results_folder, str(datetime.now()))
# backupfname = backupfname.replace(' ','_')
# backup_code(backupfname, CAFFE_DIR+'/python/', ['.py', '.prototxt'], ['result', 'log',])


print
print '........'
print 'PARAMETERS'
print '........'
print

if options.run_mode == 1:
    use_lowrank = 1
else:
    use_lowrank = 0

lowrank_weight = 1
softmax_weight = 1
N = 512
batch_size_train = 256
batch_size_test = 100
max_iter = 18000 # 24000

  ################################################################################
##### DEFINE CLASSES TO USE
if options.dataset == 'cifar100':
  num_classes = 100
  source_train = '/home/jlezama/code/caffe/data/cifar100/cifar100_train_lmdb'
  source_test = '/home/jlezama/code/caffe/data/cifar100/cifar100_test_lmdb'
  batch_size_train = 164
  mean_image_filename = '/home/jlezama/code/caffe/data/cifar100/mean.binaryproto'

else:
  num_classes = 10
  source_train = '/home/jose/Documents/code/caffe3/examples/cifar10/cifar10_train_lmdb'
  source_test = '/home/jose/Documents/code/caffe3/examples/cifar10/cifar10_test_lmdb'
  mean_image_filename = '/home/jose/Documents/code/caffe3/examples/cifar10/mean.binaryproto'
  


################################################################################
##### SAVE PARAMETERS
all_parameters = dict( (name,eval(name)) for name in ['batch_size_train',  'use_lowrank', 'softmax_weight', 'lowrank_weight', 'source_train', 'source_test',  'N', 'num_classes', ])



for k,v in all_parameters.iteritems():
   print k, ' = ',  ('\'%s\''%v if isinstance(v, str) else v)







################################################################################
### CREATE NETWORKS

outfname_train =  '%s/train2.prototxt' % (results_folder)
outfname_deploy = '%s/train2.prototxt' % (results_folder)
outfname_solver = '%s/solver.prototxt' % (results_folder)
snapshot_dir =    '%s/' % (results_folder)


print
print '........'
print 'CREATING NETWORK'
print '........'
print


# create_lmsm_network3(outfname_train, outfname_deploy, source_train, source_test, softmax_weight, lowrank_weight,  use_lowrank, batch_size_train, N, num_classes,  mu=1.)
# create_lmsm_solver3(outfname_solver, outfname_train, snapshot_dir=snapshot_dir)

  

# ################################################################################
# ### TRAIN NETWORK
# weights_str = '' #'-weights %s/models/cifar10_nin.caffemodel' % CAFFE_DIR

# command = "%s/build/tools/caffe train -solver %s %s %s  2>&1 | grep -v Restarting " % (CAFFE_DIR, outfname_solver, caffe_gpu_command_str, weights_str)

  
# if PRINT_TO_FILE:
#   command += " | tee %s/trainlog.txt" % ( results_folder)

# os.system(command)



################################################################################
# load network
model = outfname_deploy
weights = '%s/_iter_%i.caffemodel' % (snapshot_dir, max_iter)

net = caffe.Net(model, weights, caffe.TEST)



print
print '........'
print 'VISUALIZING RESULTS'
print '........'
print


mydata = load_train_test_lmdb(source_train, source_test) # load only test data

mean_image = load_mean_image(mean_image_filename)

# # extract features and w
# features1 = low_rank_map_data(net, mydata['X_val'],  256, 'bn_ip', mean_image=mean_image);
# w = net.params['ip2'][0].data[...]

# iterate through dataset
# mX_train = low_rank_map_data(net, mydata['X_train'], num_classes, 'fc8_facescrub');
features, mX_val, y = map_data_fixed(net, 'ip2')


ymine = np.argmax(mX_val,1)
acc = np.sum(ymine==y.astype(int))/np.float(y.shape[0])
print 'accuracy %2.6f' % acc


u, s, v = np.linalg.svd(features)

s= s/np.max(s)
for i in range(20):
  print i, s[i]


# # load training
net = caffe.Net(model, weights, caffe.TRAIN)
features_train, mX_train, y_train = map_data_fixed(net, 'ip2', 50000/batch_size_train)
#features_train, mX_train, y_train = -1,-1,-1

# u, s,v = np.linalg.svd(features_train)


# s= s/np.max(s)
# for i in range(20):
#   print i, s[i]


# # # save
scipy.io.savemat('%s/eval_result.mat' % (results_folder), mdict={'mX': mX_val, 'y':y, 'features':features, 'features_train':features_train, 'mX_train':mX_train, 'y_train':y_train})





if PRINT_TO_FILE:
  sys.stdout = orig_stdout
  logfile.close()


print '---'
print 'saved results to ', results_folder
