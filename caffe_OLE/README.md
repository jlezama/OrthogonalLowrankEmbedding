# Orthogonal Lowrank Embedding
This is the Caffe code for

```
    "OLÉ: Orthogonal Low-rank Embedding, A Plug and Play Geometric Loss for Deep Learning" 
    José Lezama, Qiang Qiu, Pablo Musé and Guillermo Sapiro
```

This code is based on [Large Margin Softmax Loss](https://github.com/wy1iu/LargeMargin_Softmax_Loss) by Weiyang Liu, Yandong Wen, Zhiding Yu and Meng Yang

### Files
- Caffe library
- OLE Loss
  * python/OLE.py
- CIFAR10  example
  * examples/OLE/cifar.py
- toy example
  * examples/test

### Usage
- The prototxt of OLE loss layer is as follows:
```
layer {
  name: "lowrank"
  type: "Python"
  bottom: "deep_feature"
  bottom: "label"
  top: "lowrank"
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
