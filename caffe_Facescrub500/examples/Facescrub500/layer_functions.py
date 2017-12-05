"""Functions that create a caffe layer and returns description for
prototxt """

def weight_filler_str(weight_std):
    if weight_std == 0:
        weight_filler = 'type: "xavier"'
    else:
        weight_filler = 'type: "gaussian" \n      std: %f' % float(weight_std)
    return weight_filler


def convolution_layer(name, bottom, top=None, lr_mult_1=1, lr_mult_2=2, num_output=16, kernel_size=3, stride=1, pad=0, weight_std=0):
    if not top:
        top=name

    weight_filler = weight_filler_str(weight_std)
    
    txt = open('templates/conv_layer.txt', 'r').read()

    txt = txt.replace('_NAME_', name)
    txt = txt.replace('_BOTTOM_', bottom)
    txt = txt.replace('_TOP_', top)
    txt = txt.replace('_LR_MULT_1_', str(lr_mult_1))
    txt = txt.replace('_LR_MULT_2_', str(lr_mult_2))
    txt = txt.replace('_NUM_OUTPUT_', str(num_output))
    txt = txt.replace('_KERNEL_SIZE_', str(kernel_size))    
    txt = txt.replace('_STRIDE_', str(stride))
    txt = txt.replace('_PAD_', str(pad))
    txt = txt.replace('_WEIGHT_FILLER_', weight_filler)

    return txt


def relu_layer(name, bottom, top=None):
    if not top:
        top = bottom
        
    txt = open('templates/relu_layer.txt', 'r').read()
    txt = txt.replace('_NAME_', name)
    txt = txt.replace('_BOTTOM_', bottom)
    txt = txt.replace('_TOP_', top)

    return txt

def fully_connected_layer(name, bottom, top=None, num_output=16, weight_std=0):
    if not top:
        top=name

    weight_filler = weight_filler_str(weight_std)
    
    txt = open('templates/fully_connected_layer.txt', 'r').read()

    txt = txt.replace('_NAME_', name)
    txt = txt.replace('_BOTTOM_', bottom)
    txt = txt.replace('_TOP_', top)
    txt = txt.replace('_NUM_OUTPUT_', str(num_output))
    txt = txt.replace('_WEIGHT_FILLER_', weight_filler)

    return txt

def pooling_layer(name, bottom, top=None, pool_type='MAX', kernel_size=2, stride=2):
    if not top:
        top=name

    txt = open('templates/pooling_layer.txt', 'r').read()

    txt = txt.replace('_NAME_', name)
    txt = txt.replace('_BOTTOM_', bottom)
    txt = txt.replace('_TOP_', top)
    txt = txt.replace('_POOL_TYPE_', pool_type)
    txt = txt.replace('_KERNEL_SIZE_', str(kernel_size))    
    txt = txt.replace('_STRIDE_', str(stride))

    return txt
        
def batchnorm_layer(name, bottom, top=None, use_global_stats=False, lr_mult=0, phase='TRAIN', deploy = False):
    if not top:
        top = name

    if deploy:
        deploy_comment = '#'
        use_global_stats = True
    else:
        deploy_comment = ''
        
    if use_global_stats:
        use_global_stats = 'true'
    else:
        use_global_stats = 'false'

        

    txt = open('templates/batchnorm_layer.txt', 'r').read()

    txt = txt.replace('_NAME_', name)
    txt = txt.replace('_BOTTOM_', bottom)
    txt = txt.replace('_TOP_', top)
    txt = txt.replace('_PHASE_', phase)
    txt = txt.replace('_USE_GLOBAL_STATS_', use_global_stats)
    txt = txt.replace('_LR_MULT_', str(lr_mult))    
    txt = txt.replace('_DEPLOY_COMMENT_', deploy_comment)

    return txt


def scale_layer(name, bottom, top=None, power=1, scale=1, shift=0):
    
    if not top:
        top=name

    txt = open('templates/scale_layer.txt', 'r').read()

    txt = txt.replace('_NAME_', name)
    txt = txt.replace('_BOTTOM_', bottom)

    txt = txt.replace('_TOP_', top)
    txt = txt.replace('_POWER_', str(power))
    txt = txt.replace('_SCALE_', str(scale))    
    txt = txt.replace('_SHIFT_', str(shift))

    return txt



def softmax_layer(name, bottom, label='label', loss_weight=1):
    
    txt = open('templates/softmax_layer.txt', 'r').read()
    txt = txt.replace('_NAME_', name)
    txt = txt.replace('_BOTTOM_', bottom)
    txt = txt.replace('_LABEL_', label)
    txt = txt.replace('_LOSS_WEIGHT_', str(loss_weight))

    return txt

def lowrank_layer(name, bottom, label='label', loss_weight=1):
    
    txt = open('templates/lowrank_layer.txt', 'r').read()
    txt = txt.replace('_NAME_', name)
    txt = txt.replace('_BOTTOM_', bottom)
    txt = txt.replace('_LABEL_', label)
    txt = txt.replace('_LOSS_WEIGHT_', str(loss_weight))

    return txt

def data_layer(name, top='data', label='label', source_train='lmdb/train', batch_size_train=16, source_test='lmdb/test', batch_size_test=2, data_type='lmdb'):
    
    if data_type == 'lmdb':
        txt = open('templates/data_layer.txt', 'r').read()
    elif data_type == 'hdf5':
        txt = open('templates/hdf5_data_layer.txt', 'r').read()
    else:
        raise ValueError('Unknown data layer data type')

    txt = txt.replace('_NAME_', name)
    txt = txt.replace('_TOP_', top)
    txt = txt.replace('_LABEL_', label)

    txt = txt.replace('_SOURCE_TRAIN_', source_train)
    txt = txt.replace('_BATCH_SIZE_TRAIN_', str(batch_size_train))
    txt = txt.replace('_SOURCE_TEST_', source_test)
    txt = txt.replace('_BATCH_SIZE_TEST_', str(batch_size_test))
    

    return txt

def deploy_data_layer(name, input='data', input_dim_1=1, input_dim_2=3, input_dim_3=32, input_dim_4=32):
    txt = open('templates/deploy_data_layer.txt', 'r').read()
    txt = txt.replace('_NAME_', name)
    txt = txt.replace('_INPUT_', input)
    txt = txt.replace('_INPUTDIM_1_', str(input_dim_1))
    txt = txt.replace('_INPUTDIM_2_', str(input_dim_2))
    txt = txt.replace('_INPUTDIM_3_', str(input_dim_3))
    txt = txt.replace('_INPUTDIM_4_', str(input_dim_4))

    return txt



# ################################################################################
# print convolution_layer('conv1', 'data', top='test2', num_output=132, weight_std=1e-6)

# print relu_layer('relu1', 'conv1')

# print full_connected_layer('ip1', 'conv1', 'test4', num_output=122, weight_std=0.1)

# print pooling_layer('conv2', 'conv1', top='test6', pool_type='AVE', kernel_size=3, stride=1)

# print batchnorm_layer('bn1', 'bn2',  use_global_stats=1, lr_mult=1, phase='TEST', deploy=0)

# print scale_layer('scale1', 'scale_bottom', 'test 8', power=2, scale=3, shift=4)

# print softmax_layer('loss', 'ate')

# print lowrank_layer('lowrank_loss', 'test')

# print data_layer('cifarsmall', source_train='trainfile', source_test='testfile', batch_size_train=100, batch_size_test=120)

# print deploy_data_layer('cifarsmall')
