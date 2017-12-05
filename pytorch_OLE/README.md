# Orthogonal Lowrank Embedding (OL\'E)
Classification on CIFAR-10/100 with OLE using PyTorch.

Implementaion of the article
```
"OLÉ: Orthogonal Low-rank Embedding, A Plug and Play Geometric Loss for Deep Learning"

José Lezama, Qiang Qiu, Pablo Mus\é and Guillermo Sapiro
```


Borrows heavily from [https://github.com/bearpaw/pytorch-classification]

## Install
* Install [PyTorch](http://pytorch.org/)
* Clone recursively
  ```
  git clone --recursive https://github.com/jlezama/OrthogonalLowrankEmbedding.git
  ```

* Make sure scipy is linked against openblas 

## Training Examples
First cd into this folder:
```
cd pytorch_OLE
```


#### ResNet-110 on CIFAR10+
```
python cifar.py -a resnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --lambda_ 0.25
```

#### VGG19 on CIFAR100
```
python cifar.py -a vgg19_bn --dataset cifar100 --epochs 164 --schedule 81 122 --gamma 0.1 --lambda_ 0.1 --no_augment
```

#### Standard softmax loss
Set lambda_ to 0 for standard softmax loss training

#### Evaluation script
Saves deep features to .MAT file

```
python eval.py --resume path_to/checkpoint.pth.tar
```


## Results
Top1 error rate on the CIFAR-10/100 benchmarks are reported in the
paper (Table 2).
This code is used for VGG-11, VGG-19, ResNet-110, PreResNet-110 architectures.
For STL-10,  VGG-16 VGG-Face and DenseNets see [here](../)

You may get different results when training your models with different
random seed.

The value of lambda_ needs to be set via validation for every
different architecture/dataset