# Orthogonal Lowrank Embedding (OL\'E)
Classification on CIFAR-10/100 with OLE using CAFFE.

Implementaion of the article
"OL\'E: Orthogonal Low-rank Embedding, A Plug and Play Geometric Loss for Deep Learning"
Jos\'e Lezama, Qiang Qiu, Pablo Mus\'e and Guillermo Sapiro

Borrows the Caffe version from [https://github.com/wy1iu/LargeMargin_Softmax_Loss] to allow for a direct comparison.

## Install
* Install Caffe with Pycaffe
* Make sure scipy is linked against openblas 
* Download CIFAR10 data, convert to LMDB and compute mean image using standard Caffe tools.

## Quick Test
A test script for the standalone OLE loss with synthetic data is found inside the 'test' folder

```
cd test/
python test.py
```


## Training
to run with OLE loss on CIFAR 10 just do
```
python cifar.py
```

to run with standard softmax loss ddo
```
python cifar.py -l 0
```

## Visualization
Results and features in .MAT format will be saved in the 'results' folder