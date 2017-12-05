""" Python script for training OLE networks with Caffe on CIFAR-10

Implementaion of the article
"OL\'E: Orthogonal Low-rank Embedding, A Plug and Play Geometric Loss for Deep Learning"
Jos\'e Lezama, Qiang Qiu, Pablo Mus\'e and Guillermo Sapiro


 Copyright (c) Jose Lezama, 2017

 jlezama@fing.edu.uy



For my setup, I need to run this before executing:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jose/Documents/code/anaconda2/lib/
export PYTHONPATH=/home/jose/Documents/code/caffe_OLE/python/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jose/Documents/code/anaconda2/pkgs/jpeg-9b-0/lib/


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


if computer_name == 'corsair':
  CAFFE_DIR = '/home/jose/Documents/code/caffe_OLE/'
  SOLVER_MODE = 'GPU'
  os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH'] + ':/home/jose/Documents/code/anaconda2/lib/:/home/jose/Documents/code/anaconda2/pkgs/jpeg-9b-0/lib/'
  os.environ['PYTHONPATH'] = '/home/jose/Documents/code/caffe_OLE/python/'


sys.path.append('%s/python' % CAFFE_DIR)

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
parser.add_option('-p', '--print_to_file', action='store', dest='print_to_file', help='1 to save log to file, 0 to print to stdout', default=1, type=int)
parser.add_option('-l', '--lambda_', action='store', dest='lambda_', help='Weight of OLE loss. Default 0.5. Use 0 for standard softmax loss', default=0.5, type=float)



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
ROOT_DIR = '%s/examples/OLE/' % CAFFE_DIR

RESULTS_DIR = '%s/results/' % (ROOT_DIR)
os.system("mkdir -p %s" % RESULTS_DIR)
foldername = str(uuid.uuid4())
results_folder = '%s/%s' % (RESULTS_DIR, foldername)
os.system('mkdir -p %s' % results_folder)

time.sleep(3) # sometimes I must wait for directory creation
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


backupfname = '%s/caffe_pythoncode_snapshot_%s.zip' % (results_folder, str(datetime.now()))
backupfname = backupfname.replace(' ','_')
backup_code(backupfname, CAFFE_DIR+'/python/', ['.py', '.prototxt'], ['result', 'log',])


print
print '........'
print 'PARAMETERS'
print '........'
print

if options.lambda_ > 0:
    use_OLE = 1
else:
    use_OLE = 0

lambda_ = options.lambda_
softmax_weight = 1
batch_size_train = 256 #256
batch_size_test = 100 #100
max_iter =  24000

################################################################################
##### DEFINE CLASSES TO USE
num_classes = 10

# you should get your CIFAR10 samples in lmdb files like this:
source_train = '/home/jose/Documents/code/caffe3/examples/cifar10/cifar10_train_lmdb'
source_test = '/home/jose/Documents/code/caffe3/examples/cifar10/cifar10_test_lmdb'
mean_image_filename = '/home/jose/Documents/code/caffe3/examples/cifar10/mean.binaryproto'
  


################################################################################
##### SAVE PARAMETERS
all_parameters = dict( (name,eval(name)) for name in ['batch_size_train',  'use_OLE', 'softmax_weight',  'source_train', 'source_test',  'lambda_', 'num_classes', ])



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


create_lmsm_network(outfname_train, outfname_deploy, source_train, source_test, softmax_weight,  use_OLE, batch_size_train, lambda_, num_classes)

create_lmsm_solver(outfname_solver, outfname_train, max_iter=max_iter, snapshot_dir=snapshot_dir)

  

################################################################################
### TRAIN NETWORK
weights_str = ''

command = "%s/build/tools/caffe train -solver %s %s %s  2>&1 | grep -v Restarting " % (CAFFE_DIR, outfname_solver, caffe_gpu_command_str, weights_str)

  
if PRINT_TO_FILE:
  command += " | tee %s/trainlog.txt" % ( results_folder)

os.system(command)


################################################################################
# load network
model = outfname_train
weights = '%s/_iter_%i.caffemodel' % (snapshot_dir, max_iter)

net = caffe.Net(model, weights, caffe.TEST)


print
print '........'
print 'VISUALIZING RESULTS'
print '........'
print


mydata = load_train_test_lmdb(source_train, source_test) # load only test data

mean_image = load_mean_image(mean_image_filename)

features, scores, y = map_data_fixed(net, 'ip2')


ymine = np.argmax(scores,1)
acc = np.sum(ymine==y.astype(int))/np.float(y.shape[0])
print 'accuracy %2.6f' % acc

# # save
scipy.io.savemat('%s/result.mat' % (results_folder), mdict={'scores': scores, 'labels':mydata['y_val'], 'features':features, 'y':y})



if PRINT_TO_FILE:
  sys.stdout = orig_stdout
  logfile.close()


print '---'
print 'saved results to ', results_folder
