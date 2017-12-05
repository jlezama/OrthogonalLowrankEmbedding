import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
from scipy.spatial import distance
import h5py
from PIL import Image

import sys
# add caffe to path
sys.path.append('/Users/jose/Documents/duke/code/caffe/python')
sys.path.append('/home/jose/Documents/code/caffe2/python')


import caffe
import lmdb

caffe.set_mode_gpu() #for cpu mode do 'caffe.set_mode_cpu()'


########################################################################
def load_image( infilename ) :
  img = Image.open( infilename )
  img.load()
  data = np.asarray( img, dtype="int32" )
  return data
                
########################################################################
def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y


def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(cifar10_dir = 'cs231n/datasets/cifar-10-batches-py', num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    # cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    
    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }


def load_CIFAR10_small_data(cifar10_dir, save_folder = 'data', classes=None):
    """ loads a subset of CIFAR10 classes """ 
    if classes is None:
        classes = np.array([0,1,2])
        classes_string = ''

    #else:
    classes_string = '_' + '_'.join([str(x) for x in np.sort(classes).tolist()])

    # create folder
    try:
      os.mkdir(save_folder)
    except:
      pass
    
    # check if files already produced
    X_train_filename = 'X_train%s.npy' % classes_string
    X_test_filename = 'X_test%s.npy' % classes_string
    X_val_filename = 'Xval%s.npy' % classes_string
    y_train_filename = 'y_train%s.npy' % classes_string
    y_test_filename = 'y_test%s.npy' % classes_string
    y_val_filename = 'y_val%s.npy' % classes_string
    
    if not os.path.isfile(os.path.join(save_folder, X_train_filename)):
        print "generating data"
        data = get_CIFAR10_data(cifar10_dir)
              
        # define a small dataset
         
        mydata = extract_classes_from_data(data, classes)

        np.save(os.path.join(save_folder, X_train_filename), mydata['X_train'])
        np.save(os.path.join(save_folder, X_test_filename),  mydata['X_test'])
        np.save(os.path.join(save_folder, X_val_filename),   mydata['X_val'])
        np.save(os.path.join(save_folder, y_train_filename), mydata['y_train'])
        np.save(os.path.join(save_folder, y_test_filename),  mydata['y_test'])
        np.save(os.path.join(save_folder, y_val_filename),   mydata['y_val'])
    
    else:
        print "data already generated"
        mydata={}
        print os.path.join(save_folder, X_train_filename)
        mydata['X_train'] = np.load(os.path.join(save_folder, X_train_filename))
        mydata['X_test'] =  np.load(os.path.join(save_folder, X_test_filename))
        mydata['X_val'] =   np.load(os.path.join(save_folder, X_val_filename))
        mydata['y_train'] = np.load(os.path.join(save_folder, y_train_filename))
        mydata['y_test'] =  np.load(os.path.join(save_folder, y_test_filename))
        mydata['y_val'] =   np.load(os.path.join(save_folder, y_val_filename))
        
    return mydata


################################################################################
def extract_classes_from_data(data, classes):
        """ data must be a dictionary with keys: X_train, X_test, X_val, y_train,
        y_test, y_val """

        mydata = {}

        for c in classes:
           if not 'X_train' in mydata:
             # initialize arrays
        
             mydata['X_train'] = data['X_train'][data['y_train']==c,:]
             mydata['X_test']  = data['X_test'][data['y_test']==c,:]
             mydata['X_val'] = data['X_val'][data['y_val']==c,:]
             mydata['y_train'] = data['y_train'][data['y_train']==c]
             mydata['y_test'] = data['y_test'][data['y_test']==c]
             mydata['y_val'] = data['y_val'][data['y_val']==c]
    
           else:
             # concatenate to existing arrays
             mydata['X_train'] = np.concatenate((mydata['X_train'], data['X_train'][data['y_train']==c,:]),0)
             mydata['X_test'] = np.concatenate((mydata['X_test'], data['X_test'][data['y_test']==c,:]),0)
             mydata['X_val'] = np.concatenate((mydata['X_val'], data['X_val'][data['y_val']==c,:]),0)
             mydata['y_train'] = np.concatenate((mydata['y_train'],data['y_train'][data['y_train']==c]), 0)
             mydata['y_test'] = np.concatenate((mydata['y_test'],data['y_test'][data['y_test']==c]), 0)
             mydata['y_val'] = np.concatenate((mydata['y_val'],data['y_val'][data['y_val']==c]), 0)
        return mydata

################################################################################
def write_lmdb_file(X, y, fname = 'data/mylmdb'):
    # We need to prepare the database for the size. We'll set it 10 times
    # greater than what we theoretically need. There is little drawback to
    # setting this too big. If you still run into problem after raising
    # this, you might want to try saving fewer entries in a single
    # transaction.
    map_size = X.nbytes * 10
    
    env = lmdb.open(fname, map_size=map_size)
    

    txn =  env.begin(write=True) 
    # txn is a Transaction object
    for i in xrange(X.shape[0]):
            data = X[i]
            label = y[i]

            print data.shape, data.ravel()
            datum = caffe.io.array_to_datum(data, label)
            # keystr = '{:0>8d}'.format(i)
            keystr = '{:05}'.format(np.random.randint(1,70000))+'{:05}'.format(i) # random, jose
            txn.put( keystr, datum.SerializeToString() )
            

            # datum = caffe.proto.caffe_pb2.Datum()
            # datum.channels = X.shape[1]
            # datum.height = X.shape[2]
            # datum.width = X.shape[3]
            # datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9


            # datum.label = int(y[i])

            # str_id = '{:08}'.format(i)
    
            # # The encode is only essential in Python 3
            # txn.put(str_id, datum.SerializeToString())    

    
    txn.commit()


################################################################################
def create_lmdb(classes=None, outfolder='data',   dataset_dir = '/Users/jose/Documents/anii/code/20170116_CS321n/sandbox/cs231n/datasets/'):
    if classes is None:
        classes = np.array([0,1,2])

    classes_string = '_'.join([str(x) for x in np.sort(classes).tolist()])
  
    os.system("mkdir -p %s" % outfolder)
    
    mydata = load_CIFAR10_small_data(cifar10_dir = os.path.join(dataset_dir, 'cifar-10-batches-py'), save_folder = os.path.join(dataset_dir, 'data'), classes=classes)
    
    for k, v in mydata.iteritems():
      print '%s: ' % k, v.shape
    
    # write lmdb file
    for typ in ['train', 'val', 'test']:
        write_lmdb_file(mydata['X_%s' % typ], mydata['y_%s' % typ], '%s/%s_%s' % (outfolder, typ, classes_string))





################################################################################
def low_rank_map_data(net, X, num_classes=3, layer='affine', mean_image=0):
  """Runs data X through lowrank network 'net', which is expected to
  have its last layer named as argument layer """



  # assert layer in net.blobs

  N = X.shape[0]
  mX = np.zeros((N,num_classes))


  for i in xrange(N):
       # print X[i].shape
       myimage = (X[i]-mean_image)#.transpose([0, 2, 1])
       # print 'myimage', myimage[0,0,...]
       # print net.blobs['data'].data.shape
       net.blobs['data'].data[0,...] = myimage
       #print 'blob', net.blobs['bn_ip'].data[0,0,...]
       out = net.forward( end=layer)
       output = out[layer][0]
       #print 'output', output[0,...]
       #print net.blobs

       if np.any(np.isnan(output)):
         print 'found NaN values in sample %i, skipping...' % i
         print 'ouput was', output[0:10]
         print 'input was', net.blobs['data'].data[...][0:10]
         continue
       mX[i,:] = output



  return mX

################################################################################
def compute_angles(mX, y, classes=None):
    """computes angle scores between subspaces in mX with classes y  (only three classes)
    returns the average angle and a string with description of angles for each group
    """


    if classes is None:
      classes = np.array([0,1,2])



    num_classes = classes.size

    norms = np.sqrt(np.sum(mX**2,axis=1))

    # compute angles between resulting subspaces
    angles_str = ''
    angles_mean = np.zeros(3)
    angles_count = 0
    last_angles = np.array([0, 0, 0])
    
    mult = 3.
    for val in range(int(-8*mult),int(8*mult)):
            val = val/mult
            normThresh = 10**(-val)
    
            normcount =  np.sum(norms<normThresh)
    
            # print 'number of points with norm < %f: %i/%i' % (normThresh, normcount, mX.shape[0])
            # evaluate angles for small samples (0.01)
           
            if normcount <5*num_classes:
                angles_str += '\n samples (<%f (%i/%i))' % (normThresh, normcount, mX.shape[0])
    
                break
            mX2 = mX[norms<normThresh,:]
            y2 = y[norms<normThresh]
            [U0,S0,V0] = np.linalg.svd(mX2[y2==classes[0]].T)
            [U1,S1,V1] = np.linalg.svd(mX2[y2==classes[1]].T)
            [U2,S2,V2] = np.linalg.svd(mX2[y2==classes[2]].T)
    
    
            angle201 = np.arccos(np.dot(U0[:,0],U1[:,0]))*180/np.pi
            angle202 = np.arccos(np.dot(U0[:,0],U2[:,0]))*180/np.pi
            angle212 = np.arccos(np.dot(U1[:,0],U2[:,0]))*180/np.pi
    
    
            current_angles = np.array([[angle201, angle202, angle212]])
            current_angles = np.concatenate((current_angles,180-current_angles),axis=0)
            current_angles = np.min( current_angles, axis=0)
            if np.linalg.norm(current_angles-last_angles)>1e-6:
                # not the same angles:
                angles_mean += current_angles
                angles_count+=1
    
                angles_str += '\nAngles for samples (<%f (%i/%i)): 0-1: %f,  0-2: %f,  1-2: %f' % (normThresh, normcount, mX.shape[0],angle201, angle202, angle212)
    
            last_angles = current_angles

    
    average_angles =  np.mean(angles_mean)/np.float(angles_count)

    return average_angles, angles_count, angles_str

################################################################################
def compute_distance_accuracy(X_val, X_train, y_val, y_train, metric='euclidean'):
    """computes NN classification accuracy between Xval and Xtrain using metric (euclidean or
cosine) """


    dst = distance.cdist(X_val,X_train, metric=metric)
    y_pred = y_train[np.argmin(dst,axis=1)]
    return np.sum(y_pred==y_val)/np.float(y_val.shape[0])




################################################################################
def example_network(batch_size, fname='cifar_network.prototxt'):
    hid = 100
    std = 1e-3

    n = caffe.NetSpec()

    n.data, n.label = L.DummyData(shape=[dict(dim=[batch_size, 3]),
                                         dict(dim=[batch_size])],
                                  transform_param=dict(scale=1.0/255.0),
                                  ntop=2)

    n.ip1 = L.InnerProduct(n.data, num_output= hid, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)

    n.ip2 = L.InnerProduct(n.ip1, num_output= hid, weight_filler=dict(type='xavier'))
    n.relu2 = L.ReLU(n.ip2, in_place=True)

    n.ip3 = L.InnerProduct(n.ip2, num_output= hid, weight_filler=dict(type='xavier'))
    n.relu3 = L.ReLU(n.ip3, in_place=True)


    n.affine = L.InnerProduct(n.relu3, num_output=3)
    n.lowrank = L.Python(n.affine, n.label,
                          python_param=dict(
                                          module='LowRankLoss',
                                          layer='LowRankLossLayer'),
                                          ntop=1,)
                                          #param_str='{ "param_name": param_value }'),

    f =  open(fname, 'w')
    txt = str(n.to_proto())

    
    f.write(txt)
    f.close()


################################################################################
def load_train_test_lmdb(source_train, source_test):
  mydata = {}
  X_val, y_val = load_lmdb(source_test)
  X_train, y_train = load_lmdb(source_train)
  mydata['X_train'] = X_train
  mydata['X_val'] = X_val
  mydata['y_train'] = y_train
  mydata['y_val'] = y_val



  return mydata

################################################################################
def load_mean_image(mean_image_filename):
  blob = caffe.proto.caffe_pb2.BlobProto()
  data = open(mean_image_filename , 'rb' ).read()
  blob.ParseFromString(data)
  arr = np.array( caffe.io.blobproto_to_array(blob) )
  out = arr[0]
  return out


################################################################################
def load_lmdb(lmdbfolder):
  """ loads X and y from lmdb """


  lmdb_env = lmdb.open(lmdbfolder)
  lmdb_txn = lmdb_env.begin()
  lmdb_cursor = lmdb_txn.cursor()
  datum = caffe.proto.caffe_pb2.Datum()


  N = lmdb_env.stat()['entries']

  

  count = 0
  
  for key, value in lmdb_cursor:
    datum.ParseFromString(value)
    yi = np.asarray([datum.label,])
    Xi = caffe.io.datum_to_array(datum)

    if not 'X' in locals():
        # X stores final data, but size depends on dataset

      X = np.zeros((N,Xi.shape[0], Xi.shape[1], Xi.shape[2]))
      y = np.ones(N)*-1

    # print Xi.shape, X.shape

    # else:
    #   X = np.concatenate((X, Xi), axis=0)
    #   y = np.concatenate((y, yi), axis=0)

    X[count,:] = Xi
    y[count] = yi


    # print 'reading lmdb %s %i' % (lmdbfolder, count)
    count+=1
    

  return X, y


################################################################################
def load_wiki_dataset(trainlmdb, testlmdb):
  """ loads cross-modal dataset wiki from LMDB """

  print 'reading WIKI lmdb data from %s and %s' % (trainlmdb, testlmdb)

  X_train, y_train = load_lmdb(trainlmdb)
  X_val, y_val = load_lmdb(testlmdb)
    
  data = {'X_train': X_train, 'y_train': y_train.astype(int), 'X_val': X_val, 'y_val': y_val.astype(int)}

  return data

################################################################################
def hdf5_read(hdf5_fname):
  f = h5py.File(hdf5_fname, 'r')
  
  # List all groups
  # print("Keys: %s" % f.keys())
  a_group_key = f.keys()[0]
  
  # Get the data
  X = np.asarray(f['data'])
  y = np.asarray(f['label']).astype(int).ravel()

  return X, y

################################################################################
def read_hdf5_dataset(traintxt, testtxt):
  hdf5_train = open(traintxt, 'r').read()
  hdf5_test = open(testtxt, 'r').read()

  X_train, y_train = hdf5_read(hdf5_train)
  X_test, y_test = hdf5_read(hdf5_test)

  mydata = {}

  mydata['X_train'] = X_train
  mydata['y_train'] = y_train
  mydata['X_val'] = X_test
  mydata['y_val'] = y_test

  return mydata

#########################################################
def read_images(filelist):

  N = len(filelist)
  X = np.ones((N,3,224,224)) # None
  y = np.ones(N)*-1 # None


  
  count = 0
  for line in filelist:
    line = line.split(' ')
    filename = line[0]
    yi = np.asarray(int(line[1])).reshape((1,1))

    print filename, yi
    
    Xi = load_image(filename).transpose([2, 1, 0]).reshape(1, 3, 224, 224)

    X[count,...] = Xi
    y[count] = yi

    count+=1

    # if X == None:
    #   X = Xi
    #   y = yi
    # else:
    #   X = np.concatenate((X, Xi), 0)
    #   y = np.concatenate((y, yi), 0)


  return X,y
  
################################################################################
def read_image_data(source_test, num_classes=20):
  """ reads images from a caffe-like file (ONLY TEST)"""


  
  # X_train_savefname = 'datasets/faces_X_train_%i.npy' % num_classes;
  # y_train_savefname = 'datasets/faces_y_train_%i.npy' % num_classes;
  X_test_savefname = 'datasets/faces_X_test_%i.npy'   % num_classes;
  y_test_savefname = 'datasets/faces_y_test_%i.npy'   % num_classes;
  if not os.path.isfile(X_test_savefname):
      #trainlist = open(source_train, 'r').readlines()
      testlist = open(source_test, 'r').readlines()
      # X_train, y_train = read_images(trainlist)  
      X_test, y_test = read_images(testlist)  
        

      # np.save(X_train_savefname, X_train)
      # np.save(y_train_savefname, y_train)
      np.save(X_test_savefname, X_test)
      np.save(y_test_savefname, y_test)
  else:
    # X_train = np.load(X_train_savefname)
    # y_train = np.load(y_train_savefname)
    X_test = np.load(X_test_savefname)
    y_test = np.load(y_test_savefname)


  mydata = {}
  # mydata['X_train'] = X_train
  mydata['X_val'] = X_test
  # mydata['y_train'] = y_train
  mydata['y_val'] = y_test

    
  return mydata
  
################################################################################

def softmax(x):
  """Compute softmax values for each sets of scores in x."""
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0) # only difference
################################################################################


################################################################################
def map_data_fixed(net,  layer='affine', N=100, score_layer='ip2'):
  """Runs data X through lowrank network 'net', which is expected to
  have its last layer named as argument layer """


  features1 = None
  for i in xrange(N):
    out = net.forward()
    if features1 is None:
      features1 = net.blobs[layer].data
      mX = net.blobs[score_layer].data
      labels = net.blobs['label'].data
    else:
      features1 = np.concatenate((features1, net.blobs[layer].data), 0)
      mX = np.concatenate((mX, net.blobs[score_layer].data), 0)
      labels = np.concatenate((labels, net.blobs['label'].data), 0)
  return features1, mX, labels



################################################################################
# TEST
# source_train = '/home/jlezama/datasets/20170504_faces/facescrube_C00/list_20.txt'
# source_test = '/home/jlezama/datasets/20170504_faces/facescrube_C00/list_1.txt'
# read_image_data(source_train, source_test)

# source_train = '/home/jlezama/datasets/20170504_faces/facescrube_C00/list_train_500.txt'
# source_test = '/home/jlezama/datasets/20170504_faces/facescrube_C00/all_test.txt'
# read_image_data(source_test, 500)

# CAFFE_DIR = '/Users/jose/Documents/duke/code/caffe/'
# ROOT_DIR = '%s/examples/20170321_deepLRT2/' % CAFFE_DIR
# data_folder = '%s/datasets/mnist' % ROOT_DIR
# source_train = '%s/mnist_train.txt' % (data_folder) # hdf5
# source_test = '%s/mnist_test.txt' % (data_folder) # hdf5
# 
# mydata = read_hdf5_dataset(source_train, source_test)

## mydata = load_wiki_dataset('/Users/jose/Documents/duke/code/caffe//examples/20170321_deepLRT2/datasets/wiki/image_train', '/Users/jose/Documents/duke/code/caffe//examples/20170321_deepLRT2/datasets/wiki/image_test')


# load_lmdb('/Users/jose/Documents/duke/code/caffe/examples/20170321_deepLRT2/datasets/mnist/mnist_train_lmdb')
