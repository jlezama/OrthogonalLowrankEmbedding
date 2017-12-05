"""
Auxiliary functions to create custom networks for Caffe

Copyright (c) Jose Lezama 2017
jlezama@fing.edu.uy

"""

from layer_functions import *

def write_to_file(outfname, txt):
    f = open(outfname, 'w')
    f.write(txt)
    f.close()


def create_lmsm_network(outfname_train, outfname_deploy, source_train, source_test, softmax_weight, use_OLE, batch_size_train, lambda_, num_classes=10):
  # network from large-margin-softmax paper  

  template_train = open('model/cifar_network.prototxt', 'r').read()
  template_deploy = open('model/cifar_network.prototxt', 'r').read()

  if use_OLE:
      template_train = template_train.replace('_OLE_COMMENT_', '')
  else:
      template_train = template_train.replace('_OLE_COMMENT_', '#')
          

  template_train = template_train.replace('_BATCH_SIZE_TRAIN_', str(batch_size_train))
  
  template_train = template_train.replace('_SOURCE_TRAIN_', source_train)
  template_train = template_train.replace('_SOURCE_TEST_', source_test)

  template_train = template_train.replace('_SOFTMAX_WEIGHT_', str(softmax_weight))
  template_train = template_train.replace('_LAMBDA_', str(lambda_))
                                                             
  write_to_file(outfname_train, template_train)
  write_to_file(outfname_deploy, template_deploy)

    

############
def create_lmsm_solver(outfname, net_name, max_iter=10000, lr=0.1, weight_decay=0.0005, snapshot_dir="snapshots",  solver_mode="GPU"):

    txt = open('model/cifar_solver.prototxt', 'r').read()
        

    txt = txt.replace('_NET_NAME_', net_name)
    txt = txt.replace('_MAX_ITER_', str(max_iter))
    txt = txt.replace('_LR_', str(lr))
    txt = txt.replace('_WEIGHT_DECAY_', str(weight_decay))
    txt = txt.replace('_SNAPSHOT_DIR_', snapshot_dir)
    txt = txt.replace('_SOLVER_MODE_', solver_mode)

    write_to_file(outfname, txt)

  
  



  
def create_network(outfname_train, outfname_deploy, N_conv_layers=3, N_fully_connected_layers=3, batch_size_train=100,batch_size_test=100, source_train='datatrain', source_test='datatest', num_output_conv=32, kernel_size=3, weight_std_conv=0.01,  activation='relu', num_output_fully_connected=64, weight_std_fully_connected=0.01, do_batchnorm=1, do_last_batchnorm=1, scale=1,shift=0, weight_std_affine=0, use_softmax=0, num_classes=3, input_dim_1=1,input_dim_2=3, input_dim_3=32, input_dim_4=32, use_lowrank=1, T_dimension=None, softmax_weight=1, lowrank_weight=1, data_type='lmdb'):
    """ creates a network prototxt for training and one for deploy"""

    if T_dimension==None:
        T_dimension = num_classes
    
    train_txt = ""
    deploy_txt = ""

    train_txt += data_layer(name='data_layer', source_train=source_train, batch_size_train=batch_size_train, source_test=source_test, batch_size_test=batch_size_test, data_type=data_type)

    deploy_txt += deploy_data_layer(name='data_layer', input_dim_1=input_dim_1, input_dim_2=input_dim_2, input_dim_3=input_dim_3, input_dim_4=input_dim_4)

    last_name = 'data'

    ####### CONVOLUTIONAL LAYERS
    for i in range(N_conv_layers):
        conv_name = 'conv%i' % (i+1)
        top = conv_name

        conv_txt =  convolution_layer(conv_name, last_name, num_output=num_output_conv, kernel_size=kernel_size, weight_std=weight_std_conv)

        train_txt += conv_txt
        deploy_txt += conv_txt
        
        if activation == 'pool':
            pool_name = 'pool%i' % (i+1)
            activation_txt = pooling_layer(pool_name, conv_name)
            last_name = pool_name
        elif activation == 'relu':
            relu_name = 'relu%i' % (i+1)
            activation_txt = relu_layer(relu_name, conv_name)
            last_name = conv_name
        else:
            raise Exception('Unknown activation')
        

        train_txt += activation_txt
        deploy_txt += activation_txt

        

    ####### FULLY CONNECTED LAYERS
    for i in range(N_fully_connected_layers):
        fully_connected_name = 'ip%i' % (i+1)

        fully_connected_txt = fully_connected_layer(fully_connected_name, last_name, num_output=num_output_fully_connected, weight_std=weight_std_fully_connected)

        relu_name = 'iprelu%i' % (i+1)
        relu_txt = relu_layer(relu_name, fully_connected_name)

        batchnorm_name = 'ipbn%i' % (i+1)

        if do_batchnorm and i<N_fully_connected_layers-1:
            batchnorm_txt_train = batchnorm_layer(batchnorm_name, fully_connected_name, use_global_stats=False, phase='TRAIN', deploy=False)
            batchnorm_txt_test = batchnorm_layer(batchnorm_name, fully_connected_name, use_global_stats=True, phase='TEST', deploy=False)
            
            batchnorm_txt_deploy = batchnorm_layer(batchnorm_name, fully_connected_name, deploy=True)
            scale_txt = ''
            
            last_name = batchnorm_name
            
        elif do_last_batchnorm:
            batchnorm_txt_train = batchnorm_layer(batchnorm_name, fully_connected_name, use_global_stats=False, phase='TRAIN', deploy=False)
            batchnorm_txt_test = batchnorm_layer(batchnorm_name, fully_connected_name, use_global_stats=True, phase='TEST', deploy=False)
            
            batchnorm_txt_deploy = batchnorm_layer(batchnorm_name, fully_connected_name, deploy=True)
            scale_name = 'ipbnscaled%i' % (i+1)

            scale_txt = scale_layer(scale_name, batchnorm_name, scale=scale,shift=shift)
            
            last_name = scale_name
        else:
            batchnorm_txt_train = ''
            batchnorm_txt_test = ''
            batchnorm_txt_deploy = ''
            last_name = fully_connected_name
            scale_txt = ''
            
        train_txt += fully_connected_txt + relu_txt + batchnorm_txt_train + batchnorm_txt_test + scale_txt
        deploy_txt += fully_connected_txt + relu_txt + batchnorm_txt_deploy + scale_txt
        




    # add affine layer on top of funnel layer 
    affine_name = 'affine' # (matrix T)
    affine_txt = fully_connected_layer(affine_name, last_name, num_output=T_dimension, weight_std=weight_std_affine)

    train_txt += affine_txt
    deploy_txt += affine_txt
    
    # apply lowrank loss to output of 'affine' layer [conv - fully_connected -
    # funnel - affine - lowrank] the lowrank output is located in affine. The
    # 'funnel' layer is used to allow softmax to separate between classes before
    # LRT
    if use_lowrank:
        lowrank_txt = lowrank_layer('lowrank_loss', affine_name, loss_weight=lowrank_weight)
        train_txt += lowrank_txt

    if use_softmax:
        # apply softmax loss to output of funnel layer [conv - fully_connected - funnel - softmax]
        # add one affine layer to reduce from num_output_fully_connected to num_classes

        # apr 4. trying on top of fully connected layer
        funnel_name = 'funnel'
        funnel_txt = fully_connected_layer(funnel_name, last_name, num_output=num_classes, weight_std=weight_std_fully_connected)

        train_txt += funnel_txt
        deploy_txt += funnel_txt

        softmax_txt = softmax_layer('softmax_loss', funnel_name, loss_weight=softmax_weight)
        train_txt += softmax_txt

    write_to_file(outfname_train, train_txt)
    write_to_file(outfname_deploy, deploy_txt)

        
    return train_txt, deploy_txt


def create_solver(outfname, net_name, max_iter=10000, lr=0.0001, weight_decay=0.0005, snapshot_dir="snapshots", optimizer="Adam", solver_mode="GPU"):
    txt = open('templates/solver.txt', 'r').read()
    txt = txt.replace('_NET_NAME_', net_name)
    txt = txt.replace('_MAX_ITER_', str(max_iter))
    txt = txt.replace('_LR_', str(lr))
    txt = txt.replace('_WEIGHT_DECAY_', str(weight_decay))
    txt = txt.replace('_SNAPSHOT_DIR_', snapshot_dir)
    txt = txt.replace('_OPTIMIZER_', optimizer)
    txt = txt.replace('_SOLVER_MODE_', solver_mode)

    write_to_file(outfname, txt)


