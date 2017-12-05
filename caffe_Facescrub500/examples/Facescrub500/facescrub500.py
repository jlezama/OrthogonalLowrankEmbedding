""" 
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

from collections import OrderedDict
################################################################################
# START IMPORT CAFFE

# set some useful environmental variables according to host computer
computer_name =  os.uname()[1].split('.')[0]
print computer_name


if computer_name == 'corsair':
  CAFFE_DIR = '/home/jose/Documents/code/OrthogonalLowrankEmbedding/caffe_Facescrub500/'
  SOLVER_MODE = 'GPU'
      

  
sys.path.append('%s/python' % CAFFE_DIR)
os.environ['PYTHONPATH'] = '%s/python' % CAFFE_DIR

import caffe


# END IMPORT CAFFE
################################################################################

from auxfunctions import *
from backup import *
from network_functions import *

import uuid



################################################################################
# GET ARGUMENTS
# GPU ID

parser = optparse.OptionParser()
parser.add_option('-g', '--gpu_id', action='store', dest='gpu_id', help='ID of GPU device', default=0, type=int)
parser.add_option('-d', '--dataset', action='store', dest='dataset', help='dataset to be used (facescrub*', default="facescrub", type=str)
parser.add_option('-p', '--print_to_file', action='store', dest='print_to_file', help='1 to save log to file, 0 to print to stdout', default=1, type=int)
parser.add_option('-m', '--model', action='store', dest='model', help='network model to be used (custom*, allcnn)', default="custom", type=str)
parser.add_option('-w', '--weights', action='store', dest='weights', help='pretrained weights', default=None, type=str)
parser.add_option('-l', '--lambda_', action='store', dest='lambda_', help='\lambda, weight of OLE loss (default 250)', default=250., type=float)


options, args = parser.parse_args()

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
ROOT_DIR = '%s/examples/Facescrub500/' % CAFFE_DIR

RESULTS_DIR = '%s/results_%s' % (ROOT_DIR, options.dataset)
os.system("mkdir -p %s" % RESULTS_DIR)
foldername = str(uuid.uuid4())
results_folder = '%s/%s' % (RESULTS_DIR, foldername)
os.system('mkdir -p %s' % results_folder)

time.sleep(5) # wait for directory creation
print 'folder is ', foldername

# use sys.stdout to log to file
orig_stdout = sys.stdout

PRINT_TO_FILE = options.print_to_file


################################################################################
# MAIN

if PRINT_TO_FILE:
  logfile = file('%s/log.txt' % (results_folder), 'w')
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

backupfname = '%s/code_snapshot_%s.zip' % (results_folder, str(datetime.now()))
backupfname = backupfname.replace(' ','_')

backup_code(backupfname, '.', ['.py', '.prototxt'], ['result', 'log',])


print
print '........'
print 'PARAMETERS'
print '........'
print

# fixed
# for network
fc7_length  =  4096

batch_size_test  =  2
batch_size_train  =  26 
lambda_  = options.lambda_

max_iter  =  20000
lr  =  1e-5
lr_mult  =  1
weight_decay  =  0.001
num_classes  =  500
shuffle_train = 1

fc9_length = 1024 # extra layer

use_meanfile = 0 #  better without mean image
    
if fc7_length == 4096:
  fc7_name = 'fc7';
else:
  fc7_name = 'fc7x';

if lambda_ > 0:
  use_OLE = 1
else:
  use_OLE = 0


if fc9_length > 0:
  OLE_bottom = 'fc9'
else:
  OLE_bottom = 'fc8_facescrub'


################################################################################
##### DEFINE CLASSES TO USE

if options.dataset == 'facescrub':
  if num_classes == 500:

    ### LMDB
    if shuffle_train:
      source_train = '%s/data/facescrub_train_500_shuffle_lmdb' % CAFFE_DIR
    else:
      source_train = '%s/data/facescrub_train_500_lmdb/' % CAFFE_DIR

    source_test = '%s/data/facescrub_test_lmdb' % CAFFE_DIR # contains all identities
    source_test_500 =  '%s/data/facescrub_test_500_lmdb' % CAFFE_DIR # contains only 500 identitites

    # # text files
    # source_train = '/data/20170526_facescrub/Facescrub500/Facescrub500_train.txt'
    # source_test_500 = '/data/20170526_facescrub/Facescrub500/Facescrub500_test_500.txt'
    # source_test_all = '/data/20170526_facescrub/Facescrub500/Facescrub500_test.txt'
    
    test_iter = 5649
    
  options.weights = os.path.join(ROOT_DIR, 'models/VGG_FACE.caffemodel')
else:
  raise ValueError('Unknown Dataset: %s', options.dataset)


################################################################################
##### SAVE PARAMETERS
all_parameters = dict( (name,eval(name)) for name in ['batch_size_train','batch_size_test', 'max_iter',  'lr', 'weight_decay',  'use_OLE', 'lambda_', 'source_train', 'source_test', 'source_test_500',  'num_classes', 'lr_mult', 'fc7_length', 'fc9_length', 'shuffle_train', 'test_iter', 'OLE_bottom', 'use_meanfile'])



for k,v in all_parameters.iteritems():
   print k, ' = ',  ('\'%s\''%v if isinstance(v, str) else v)







################################################################################
### CREATE NETWORKS

outfname_train =  '%s/train.prototxt' % (results_folder)
outfname_deploy = '%s/deploy.prototxt' % (results_folder)
outfname_solver = '%s/solver.prototxt' % (results_folder)
snapshot_dir =    '%s/' % (results_folder)


print
print '........'
print 'CREATING NETWORK'
print '........'
print

create_vggface_lmdb_network(outfname_train=outfname_train, outfname_deploy=outfname_deploy, source_train=source_train, source_test=source_test_500, num_classes=num_classes, batch_size_train=batch_size_train, batch_size_test=batch_size_test, use_OLE=use_OLE, lambda_=lambda_, lr_mult=lr_mult, fc7_length=fc7_length, fc9_length=fc9_length, shuffle_train=shuffle_train, OLE_bottom=OLE_bottom, use_meanfile=use_meanfile)

create_vggface_solver(outfname=outfname_solver, net_name=outfname_train, max_iter=max_iter, lr=lr, weight_decay=weight_decay, snapshot_dir=snapshot_dir, test_iter=test_iter)


  

################################################################################
### TRAIN NETWORK
if options.weights != None:
  weights_str = '-weights %s' % options.weights
else:
  weights_str = ''

command = "%s/build/tools/caffe train -solver %s %s %s  2>&1 | grep -v Restarting " % (CAFFE_DIR, outfname_solver, caffe_gpu_command_str, weights_str)

  
if PRINT_TO_FILE:
  command += " | tee %s/trainlog.txt" % ( results_folder)

os.system(command)



################################################################################
# load network
model = outfname_deploy
weights = '%s/_iter_%i.caffemodel' % (snapshot_dir, max_iter)

net = caffe.Net(model, weights, caffe.TRAIN)

# check norm(T)

if 'fc8_facescrub' in  net.params:
  print 'norm(T)', np.linalg.norm(net.params['fc8_facescrub'][0].data[...])
  print 'T shape', net.params['fc8_facescrub'][0].data[...].shape


print
print '........'
print 'VISUALIZING RESULTS'
print '........'
print


if PRINT_TO_FILE:
  sys.stdout = orig_stdout
  logfile.close()

print '---'
print 'saved results to ', results_folder
