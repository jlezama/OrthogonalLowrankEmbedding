# Orthogonal Lowrank Embedding

This is the Caffe code for running the Facescrub 500 experiment in

```
    "OLÉ: Orthogonal Low-rank Embedding, A Plug and Play Geometric Loss for Deep Learning" 
   José Lezama, Qiang Qiu, Pablo Musé and Guillermo Sapiro
   [https://arxiv.org/abs/1712.01727](https://arxiv.org/abs/1712.01727)
```

### Files
- Caffe library
- OLE Loss
  * python/OLE.py
- Facescrub500  example
  * examples/OLE/cifar.py

### Facescrub 500 dataset
Download the Facescrub 500 dataset [here](https://iie.fing.edu.uy/~jlezama/datasets/Facescrub500/)

### Usage
- Compile this Caffe version with pycaffe enabled
- Download VGG_FACE.caffemodel from [VGG website](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/) into examples/Facescrub500/models/
- Get Facescrub 500 dataset from [here](https://iie.fing.edu.uy/~jlezama/datasets/Facescrub500/)
- Create lmdb files using Caffe tool. Example
```build/tools/convert_imageset --shuffle / Facescrub500_train.txt Facescrub500_train_shuffle.lmdb```
- Save lmdb into ```data/``` folder
- Set PYTHONPATH environment variable to the Caffe python folder (where OLE.py is)
- CD into examples/Facescrub500/ and run ```pyhton facescrub500.py```
- The prototxt of OLE loss layer is as follows:

```
layer {
  name: "OLE"
  type: "Python"
  bottom: "deep_feature"
  bottom: "label"
  top: "OLE"
  python_param {
    module: "OLE"
    layer: "OLELossLayer"
    param_str: '{"lambda_": 0.25}'
  }
  loss_weight: 1
}
```



### Disclaimer
- This code is for research purpose only.

- If you have any questions, feel free to contact:
 José  Lezama jlezama@fing.edu.uy)
