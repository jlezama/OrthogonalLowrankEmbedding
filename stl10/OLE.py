""" 
OLE loss  

Copyright (c)  Jose Lezama, 2017 
jlezama@fing.edu.uy
"""

import torch
from torch.autograd import Function


class OLELoss(Function):
    def __init__(self, n_classes, lambda_=0.25):
        super(OLELoss, self).__init__()
        self.n_classes = n_classes
        self.lambda_ = lambda_
        self.dx = None

    def forward(self, X, y):
        n, d = X.shape

        lambda_ = 1.
        delta = 1.

        # gradients initialization
        obj_c = torch.zeros(1).sum()
        dx_c = torch.zeros(n, d)
        if X.is_cuda:
            obj_c, dx_c = obj_c.cuda(), dx_c.cuda()

        eig_thd = 1e-6  # threshold small eigenvalues for a better subgradient

        # compute objective and gradient for first term \sum ||TX_c||*
        for c in range(self.n_classes):
            a = X[y == c, :]
            if a.size(0) == 0:
                continue

            u, s, v = a.svd()
            nuclear = s.sum()

            if nuclear > delta:
                obj_c += nuclear

                # discard small singular values
                r = torch.sum(s < eig_thd)
                uprod = u[:, 0:u.shape[1] - r].matmul(v[:, 0:v.shape[1] - r].t())

                dx_c[y == c, :] += uprod
            else:
                obj_c += delta

        # compute objective and gradient for secon term ||TX||*
        u, s, v = X.svd()  # all classes
        obj_all = s.sum()

        r = torch.sum(s < eig_thd)
        uprod = u[:, 0:u.shape[1] - r].matmul(v[:, 0:v.shape[1] - r].t())

        dx_all = uprod

        obj = (obj_c - lambda_ * obj_all) / n * self.lambda_
        self.dx = (dx_c - lambda_ * dx_all) / n * self.lambda_

        return obj

    def backward(self, grad_output):
        return self.dx, None

