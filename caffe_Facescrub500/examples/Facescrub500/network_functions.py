from layer_functions import *

def write_to_file(outfname, txt):
    f = open(outfname, 'w')
    f.write(txt)
    f.close()


def create_vggface_solver(outfname, net_name, max_iter=10000, lr=0.0001, weight_decay=0.001, snapshot_dir="snapshots", optimizer="Adam", solver_mode="GPU", test_iter=241):
    txt = open('models/VGG_FACE_solver.template', 'r').read()
    txt = txt.replace('_NET_NAME_', net_name)
    txt = txt.replace('_MAX_ITER_', str(max_iter))
    txt = txt.replace('_LR_', str(lr))
    txt = txt.replace('_WEIGHT_DECAY_', str(weight_decay))
    txt = txt.replace('_SNAPSHOT_DIR_', snapshot_dir)
    txt = txt.replace('_OPTIMIZER_', optimizer)
    txt = txt.replace('_SOLVER_MODE_', solver_mode)
    txt = txt.replace('_TEST_ITER_', str(test_iter))

    write_to_file(outfname, txt)


def create_vggface_network(outfname_train, outfname_deploy, source_train, source_test, num_classes, batch_size_train, batch_size_test, use_OLE=1, lambda_=0.1, lr_mult=1, fc7_length=4096, fc9_length=0, shuffle_train=1, OLE_bottom='fc8_facescrub'):
  template_train = open('models/VGG_FACE_train.template', 'r').read()
  template_deploy = open('models/VGG_FACE_deploy.template', 'r').read()

  if shuffle_train:
      template_train =   template_train.replace('_SHUFFLE_TRAIN_', '')
  else:
      template_train =   template_train.replace('_SHUFFLE_TRAIN_', '# ')
      
  template_train =   template_train.replace('_NUM_CLASSES_', str(num_classes))
  template_deploy = template_deploy.replace('_NUM_CLASSES_', str(num_classes))

  template_train =   template_train.replace('_FC7_LENGTH_', str(fc7_length))
  template_deploy = template_deploy.replace('_FC7_LENGTH_', str(fc7_length))

  if fc7_length == 4096:
      fc7_name = 'fc7'
  else:
      fc7_name = 'fc7x'
      
  template_train =   template_train.replace('_FC7_NAME_', fc7_name)
  template_deploy = template_deploy.replace('_FC7_NAME_', fc7_name)

  if fc9_length>0:
      template_train =   template_train.replace('_LAST_LAYER_NAME_', 'fc9')
      template_deploy = template_deploy.replace('_LAST_LAYER_NAME_', 'fc9')
      template_train =   template_train.replace('_FC9_COMMENT_', '')
      template_deploy = template_deploy.replace('_FC9_COMMENT_', '')
      template_train =   template_train.replace('_FC9_LENGTH_', str(fc9_length))
      template_deploy = template_deploy.replace('_FC9_LENGTH_', str(fc9_length))
  else:
      template_train =   template_train.replace('_LAST_LAYER_NAME_', fc7_name)
      template_deploy = template_deploy.replace('_LAST_LAYER_NAME_', fc7_name)
      template_train =   template_train.replace('_FC9_COMMENT_', '#')
      template_deploy = template_deploy.replace('_FC9_COMMENT_', '#')
      
  
  template_train = template_train.replace('_SOURCE_TRAIN_', source_train)
  template_train = template_train.replace('_SOURCE_TEST_', source_test)

  template_train = template_train.replace('_BATCH_SIZE_TRAIN_', str(batch_size_train))
  template_train = template_train.replace('_BATCH_SIZE_TEST_',  str(batch_size_test))
  
  template_train = template_train.replace('_LR_MULT_', str(lr_mult))
  template_train = template_train.replace('_LAMBDA_', str(lambda_))

  if use_OLE:
      template_train = template_train.replace('_OLE_COMMENT_', '')
  else:
      template_train = template_train.replace('_OLE_COMMENT_', '#')
      
  template_train = template_train.replace('_OLE_BOTTOM_', OLE_bottom)
  
  write_to_file(outfname_train, template_train)
  write_to_file(outfname_deploy, template_deploy)

##############################################
def create_vggface_lmdb_network(outfname_train, outfname_deploy, source_train, source_test, num_classes, batch_size_train, batch_size_test, use_OLE=1, lambda_=0.1, lr_mult=1, fc7_length=4096, fc9_length=0, shuffle_train=1, OLE_bottom='fc8_facescrub', use_meanfile = True):
  template_train = open('models/VGG_FACE_train_lmdb.template', 'r').read()
  template_deploy = open('models/VGG_FACE_deploy.template', 'r').read()

      
  template_train =   template_train.replace('_NUM_CLASSES_', str(num_classes))
  template_deploy = template_deploy.replace('_NUM_CLASSES_', str(num_classes))

  template_train =   template_train.replace('_FC7_LENGTH_', str(fc7_length))
  template_deploy = template_deploy.replace('_FC7_LENGTH_', str(fc7_length))

  if fc7_length == 4096:
      fc7_name = 'fc7'
  else:
      fc7_name = 'fc7x'
      
  template_train =   template_train.replace('_FC7_NAME_', fc7_name)
  template_deploy = template_deploy.replace('_FC7_NAME_', fc7_name)

  if fc9_length>0:
      template_train =   template_train.replace('_LAST_LAYER_NAME_', 'fc9')
      template_deploy = template_deploy.replace('_LAST_LAYER_NAME_', 'fc9')
      template_train =   template_train.replace('_FC9_COMMENT_', '')
      template_deploy = template_deploy.replace('_FC9_COMMENT_', '')
      template_train =   template_train.replace('_FC9_LENGTH_', str(fc9_length))
      template_deploy = template_deploy.replace('_FC9_LENGTH_', str(fc9_length))
  else:
      template_train =   template_train.replace('_LAST_LAYER_NAME_', fc7_name)
      template_deploy = template_deploy.replace('_LAST_LAYER_NAME_', fc7_name)
      template_train =   template_train.replace('_FC9_COMMENT_', '#')
      template_deploy = template_deploy.replace('_FC9_COMMENT_', '#')
      
  
  template_train = template_train.replace('_SOURCE_TRAIN_', source_train)
  template_train = template_train.replace('_SOURCE_TEST_', source_test)

  template_train = template_train.replace('_BATCH_SIZE_TRAIN_', str(batch_size_train))
  template_train = template_train.replace('_BATCH_SIZE_TEST_',  str(batch_size_test))
  
  template_train = template_train.replace('_LR_MULT_', str(lr_mult))
  template_train = template_train.replace('_LAMBDA_', str(lambda_))

  if use_OLE:
      template_train = template_train.replace('_OLE_COMMENT_', '')
  else:
      template_train = template_train.replace('_OLE_COMMENT_', '#')

  if use_meanfile:
      template_train = template_train.replace('_MEAN_FILE_COMMENT_', '')
  else:
      template_train = template_train.replace('_MEAN_FILE_COMMENT_', '#')
      
  template_train = template_train.replace('_OLE_BOTTOM_', OLE_bottom)
  
  write_to_file(outfname_train, template_train)
  write_to_file(outfname_deploy, template_deploy)
##########################

