# Orthogonal Lowrank Embedding for Caffe
#
# Copyright (c) Jose Lezama, 2017
#
# jlezama@fing.edu.uy


# MAKE SURE scipy is linked against openblas

import caffe
import numpy as np
import scipy as sp



class OLELossLayer(caffe.Layer):
    """
    Computes the OLE Loss in CPU
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

        params = eval(self.param_str)
        self.lambda_ = np.float(params['lambda_'])

        
    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].shape[0] != bottom[1].shape[0]: 
            raise Exception("Inputs must have the same number of samples.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        # bottom[0] should be X
        # bottom[1] should be classes

        X = bottom[0].data
        y = bottom[1].data


        classes = np.unique(y)
        C = classes.size
        

        N, D = X.shape

        
        DELTA = 1.
        

        # gradients initialization
        Obj_c = 0
        dX_c = np.zeros((N, D))
        Obj_all = 0;
        dX_all = np.zeros((N,D))

        eigThd = 1e-6 # threshold small eigenvalues for a better subgradient

        # compute objective and gradient for first term \sum ||TX_c||*
        for c in classes:
            A = X[y==c,:]

            # intra-class SVD
            U, S, V = sp.linalg.svd(A, full_matrices = False, lapack_driver='gesvd')
            
            V = V.T
            nuclear = np.sum(S);

            ## L_c = max(DELTA, ||TY_c||_*)-DELTA
            
            if nuclear>DELTA:
              Obj_c += nuclear;
            
              # discard small singular values
              r = np.sum(S<eigThd)
              uprod = U[:,0:U.shape[1]-r].dot(V[:,0:V.shape[1]-r].T)
            
              dX_c[y==c,:] += uprod
            else:
              Obj_c+= DELTA
            
        # compute objective and gradient for secon term ||TX||*
                                 
        U, S, V = sp.linalg.svd(X, full_matrices = False, lapack_driver='gesvd')  # all classes

        V = V.T

        Obj_all = np.sum(S);

        r = np.sum(S<eigThd)

        uprod = U[:,0:U.shape[1]-r].dot(V[:,0:V.shape[1]-r].T)

        dX_all = uprod


        obj = (Obj_c  - Obj_all)/N*self.lambda_


        dX = (dX_c  - dX_all)/N*self.lambda_

        self.diff[...] = dX
        top[0].data[...] = obj


    def backward(self, top, propagate_down, bottom):
        propagate_down[1] = 0 # labels don't need backpropagation
        

        for i in range(2):
            if not propagate_down[i]:
                continue

            # # # print '----------------------------------------------'
            # # # print '------------- DOING GRADIENT UPDATE ----------'
            # # # print '----------------------------------------------'
            # # # print
            
            # # # print 'norm(self.diff)', np.linalg.norm(self.diff)
            # # # print ''
            # # # print ''
            
            bottom[i].diff[...] = self.diff 
