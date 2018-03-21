# Orthogonal Lowrank Embedding

This repository contains the source code for the experiments of the article

    "OLÉ: Orthogonal Low-rank Embedding, A Plug and Play Geometric Loss for Deep Learning" 
    José Lezama, Qiang Qiu, Pablo Musé and Guillermo Sapiro, CVPR 2018

[https://arxiv.org/abs/1712.01727](https://arxiv.org/abs/1712.01727)

If you find this work useful in your research, please consider citing:

    @inproceedings{Lezama2018OLE,
    title={OL\'E: Orthogonal Low-rank Embedding, A Plug and Play Geometric Loss for Deep Learning},
    author={Lezama, Jos\'e and Qiu, Qiang and Mus\'e, Pablo and Sapiro, Guillermo},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    year={2018}
    }

# Experiments


[STL-10-Pytorch](stl10)  Contains experiments using small training data on STL-10 database. *(This is the simplest experiment to run, I recomend starting here)*

[Cifar10-Caffe](caffe_OLE) Contains experiments on CIFAR10 using Caffe and a VGG-16 architecture

[Facescrub500-Caffe](caffe_Facescrub500) Contains experiments on face dataset Facescrub500 

[Cifar10-Pytorch](pytorch_OLE) Contains experiments on CIFAR10 and CIFAR100 for various architectures.


# Facescrub 500 dataset
Download dataset used in the paper [here](https://iie.fing.edu.uy/~jlezama/datasets/Facescrub500/)