# Orthogonal Lowrank Embedding (OLÉ) on STL-10 with Pytorch

Implementation of the article

   "OLÉ: Orthogonal Low-rank Embedding, A Plug and Play Geometric Loss for Deep Learning"
   José Lezama, Qiang Qiu, Pablo Musé and Guillermo Sapiro

Borrows heavily from [Aaron Xichen PyTorch Playground] (https://github.com/https://github.com/aaron-xichen/pytorch-playground)


- This package contains the Python source code for running the STL-10
  experiments in the paper.

- Requires Pytorch, Torchvision, Numpy and Scipy packages, and a GPU. Scipy must be linked against openblas.

### BASIC INSTRUCTIONS: 
- To try the experiment on STL-10+ with OLE (Table 2) just run:
```
python train.py
```
- To try using only standard softmax loss run:
```
python train.py --lambda_ 0
```
- For the experiment with fewer samples (Fig. 5) run:
```
python train.py --wd 1e-4 --data_augment 0 --lambda_ 0.125 --num_samples 50 
```

- For the \lambda validation experiments (Fig. 4) run:
```
python train.py --validation 1 --lambda_ 0.25
```
- Resulting logs and model will be saved to the 'results' folder

- The script eval.py can be used to extract deep features and save
  into .MAT file. The script visualize.m presents a visualization
  example (similar to Fig. 1).

- This code was tested on Ubuntu 16.04 with a GTX-1080, Python version
2.7.13 (Anaconda), Pytorch version 0.2.0_2, Numpy version 1.12.1, and
Scipy version 0.19.1. Execution time is about 5 seconds per epoch with
500 samples.
