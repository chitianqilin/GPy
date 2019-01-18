# Author Tianqi Wei

import numpy as np
from .kern import Kern
from ...core.parameterization import Param
from paramz.transformations import Logexp
from paramz.caching import Cache_this


class KFFLFM(Kern):

    """
    This kernel approximate Second order Latent Force Model using random Fourier features.
    This kernel is implemented according to the paper
    "Fast Kernel Approximations for Latent Force Models and Convolved Multiple-Output Gaussian processes."
    by Guarnizo, Cristian, and Mauricio A. Álvarez. arXiv preprint arXiv:1805.07460 (2018).
    """

    def __init__(self, input_dim, output_dim, scale=None, mass=None, spring=None, damper=None, nfs=10, fs=None,
                 active_dims=None, name='kfflfm'):

        super(KFFLFM, self).__init__(input_dim, active_dims, name)
        self.output_dim = output_dim

        if scale is None:
            scale = np.random.rand(self.output_dim)
        self.scale = Param('scale', scale, Logexp())

        if mass is None:
            mass = np.random.rand(self.output_dim)
        self.mass = Param('scale', mass, Logexp())

        if spring is None:
            spring = np.random.rand(self.output_dim)
        self.spring = Param('scale', spring, Logexp())

        if damper is None:
            damper = np.random.rand(self.output_dim)
        self.damper = Param('scale', damper, Logexp())

        self.link_parameters(self.scale, self.mass, self.spring, self.damper, self.nfs)

        # nfs is abbreviation of "number of frequency samples", which is a variable for the Random Fourier Features
        # nfs is a constant during optimisation.
        self.nfs = nfs

        # fs is abbreviation of "frequency samples", which is a vector with nfs frequency samples
        # for the Random Fourier Features. fs is constant during optimisation.
        if fs is None:
            self.fs = np.random.rand(self.nfs)
        else:
            self.fs = fs

        alpha(1) = lfmKern1.damper. / (2 * lfmKern1.mass)
        alpha(2) = lfmKern2.damper. / (2 * lfmKern2.mass)
        omega(1) = sqrt(lfmKern1.spring. / lfmKern1.mass - alpha(1) * alpha(1))
        omega(2) = sqrt(lfmKern2.spring. / lfmKern2.mass - alpha(2) * alpha(2))


    def parameters_changed(self):
        '''
        This function overrides the same name function in the grandparent class "Parameterizable", which is simply
        "pass"
        It describes the behaviours of the class when the "parameters" of a kernel are updated.
        '''
        self.GPCoregionalizedRegressionB = np.dot(self.W, self.W.T) + np.diag(self.kappa)

    def K(self, X, X2=None):
        if use_coregionalize_cython:
            return self._K_cython(X, X2)
        else:
            return self._K_numpy(X, X2)


    def _K_numpy(self, X, X2=None):
        index = np.asarray(X, dtype=np.int)
        if X2 is None:
            return self.B[index,index.T]
        else:
            index2 = np.asarray(X2, dtype=np.int)
            return self.B[index,index2.T]

    def _K_cython(self, X, X2=None):
        if X2 is None:
            return coregionalize_cython.K_symmetric(self.B, np.asarray(X, dtype=np.int64)[:,0])
        return coregionalize_cython.K_asymmetric(self.B, np.asarray(X, dtype=np.int64)[:,0], np.asarray(X2, dtype=np.int64)[:,0])


    def Kdiag(self, X):
        return np.diag(self.B)[np.asarray(X, dtype=np.int).flatten()]

    def update_gradients_full(self, dL_dK, X, X2=None):
        index = np.asarray(X, dtype=np.int)
        if X2 is None:
            index2 = index
        else:
            index2 = np.asarray(X2, dtype=np.int)

        #attempt to use cython for a nasty double indexing loop: fall back to numpy
        if use_coregionalize_cython:
            dL_dK_small = self._gradient_reduce_cython(dL_dK, index, index2)
        else:
            dL_dK_small = self._gradient_reduce_numpy(dL_dK, index, index2)


        dkappa = np.diag(dL_dK_small).copy()
        dL_dK_small += dL_dK_small.T
        dW = (self.W[:, None, :]*dL_dK_small[:, :, None]).sum(0)

        self.W.gradient = dW
        self.kappa.gradient = dkappa

    def _gradient_reduce_numpy(self, dL_dK, index, index2):
        index, index2 = index[:,0], index2[:,0]
        dL_dK_small = np.zeros_like(self.B)
        for i in range(self.output_dim):
            tmp1 = dL_dK[index==i]
            for j in range(self.output_dim):
                dL_dK_small[j,i] = tmp1[:,index2==j].sum()
        return dL_dK_small

    def _gradient_reduce_cython(self, dL_dK, index, index2):
        index, index2 = np.int64(index[:,0]), np.int64(index2[:,0])
        return coregionalize_cython.gradient_reduce(self.B.shape[0], dL_dK, index, index2)


    def update_gradients_diag(self, dL_dKdiag, X):
        index = np.asarray(X, dtype=np.int).flatten()
        dL_dKdiag_small = np.array([dL_dKdiag[index==i].sum() for i in range(self.output_dim)])
        self.W.gradient = 2.*self.W*dL_dKdiag_small[:, None]
        self.kappa.gradient = dL_dKdiag_small

    def gradients_X(self, dL_dK, X, X2=None):
        return np.zeros(X.shape)

    def gradients_X_diag(self, dL_dKdiag, X):
        return np.zeros(X.shape)
