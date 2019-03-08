# Copyright (c) 2012, James Hensman and Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import sys

from GPy.kern.src.kern import Kern
import numpy as np
from numpy.lib.scimath import sqrt as csqrt
from GPy.core.parameterization import Param
from paramz.transformations import Logexp
from GPy.kern.src.independent_outputs import index_to_slices
from . import lfm_C
from GPy.kern.src.kern import Kern
from .rbf import RBF
#from .lfm import LFMXLFM, LFMXRBF
#from .multioutput_kern import MultioutputKern
#import numpy as np
#from ...core.parameterization import Param
# from paramz.transformations import Logexp
# from .independent_outputs import index_to_slices
# import lfm_C

#from \util.config import config # for assesing whether to use cython


def cell(d0, d1):
    if d1 == 1:
        return [None for _ in range(d0)]
    else:
        return [[None for _ in range(d1)] for _ in range(d0)]


def lfmUpsilonMatrix(gamma1_p, sigma2, X, X2):
    print((gamma1_p, sigma2, X, X2))
    return lfm_C.UpsilonMatrix(gamma1_p, sigma2, X.astype(np.float64), X2.astype(np.float64))


def lfmUpsilonVector(gamma1_p, sigma2, X):
    return lfm_C.UpsilonVector(gamma1_p, sigma2, X.astype(np.float64))


def lfmGradientUpsilonMatrix(gamma1_p, sigma2, X, X2):
    return lfm_C.GradientUpsilonMatrix(gamma1_p, sigma2, X.astype(np.float64), X2.astype(np.float64))


def lfmGradientUpsilonVector(gamma1_p, sigma2, X):
    return lfm_C.GradientUpsilonVector(gamma1_p, sigma2, X.astype(np.float64))


def lfmGradientSigmaUpsilonMatrix(gamma1_p, sigma2, X, X2):
    return lfm_C.GradientSigmaUpsilonMatrix(gamma1_p, sigma2, X.astype(np.float64), X2.astype(np.float64))


def lfmGradientSigmaUpsilonVector(gamma1_p, sigma2, X):
    return lfm_C.GradientSigmaUpsilonVector(gamma1_p, sigma2, X.astype(np.float64))


def lfmComputeH3(gamma1_p, gamma1_m, sigma2, X, X2, preFactor, mode=None, term=None):
    # LFMCOMPUTEH3 Helper function for computing part of the LFM kernel.
    # FORMAT
    # DESC computes a portion of the LFM kernel.
    # ARG gamma1 : Gamma value for first system.
    # ARG gamma2 : Gamma value for second system.
    # ARG sigma2 : length scale of latent process.
    # ARG X : first time input (number of time points x 1).
    # ARG X2 : second time input (number of time points x 1).
    # ARG mode: indicates in which way the vectors X and X2 must be transposed
    # RETURN h : result of this subcomponent of the kernel for the given values.
    print('lfmComputeH3')
    print((gamma1_p, sigma2, X, X2))
    if not mode:
        if not term:
            upsilon = lfmUpsilonMatrix(gamma1_p, sigma2, X, X2)
            h = preFactor * upsilon
        else:
            upsilon = lfmUpsilonMatrix(gamma1_p, sigma2, X, X2)
            h = -preFactor[0] * upsilon + preFactor[1] * np.conj(upsilon)

    else:

        upsilon = np.hstack([lfmUpsilonMatrix(gamma1_p, sigma2, X, X2), lfmUpsilonMatrix(gamma1_m, sigma2, X, X2)])
        h = preFactor[0] * upsilon + preFactor[1] * upsilon
    return [h, upsilon]


def lfmComputeH4(gamma1_p, gamma1_m, sigma2, X, preFactor, preExp, mode=None, term=None ):
    # LFMCOMPUTEH4 Helper function for computing part of the LFM kernel.
    # FORMAT
    # DESC computes a portion of the LFM kernel.
    # ARG gamma1 : Gamma value for first system.
    # ARG gamma2 : Gamma value for second system.
    # ARG sigma2 : length scale of latent process.
    # ARG X : first time input (number of time points x 1).
    # ARG preFactor : precomputed constants
    # ARG preExp : precomputed exponentials
    # ARG mode: indicates in which way the vectors X and X2 must be transposed
    # RETURN h : result of this subcomponent of the kernel for the given values.

    if not mode:
        if not term:
            upsilon = lfmUpsilonVector(gamma1_p, sigma2, X)[:, None]
            h = np.matmul(upsilon, (preExp / preFactor[0] - np.conj(preExp) / preFactor[1]).T)

        else:
            upsilon = lfmUpsilonVector(gamma1_p, sigma2, X)[:, None]
            h = np.matmul(upsilon, (preExp / preFactor[0]).T) - np.matmul(np.conj(upsilon), (preExp/preFactor[1]).T)

    else:
        upsilon = [lfmUpsilonVector(gamma1_p, sigma2, X)[:, None], lfmUpsilonVector(gamma1_m, sigma2, X)[:, None]]
        h = np.matmul(upsilon[0], (preExp[:, 0] / preFactor[0] - preExp[:, 1] / preFactor[1]).T) \
            + np.matmul(upsilon[1] * (preExp[:, 1] / preFactor[2] - preExp[:, 0] / preFactor[3]).T)
    return [h, upsilon]


def lfmGradientH31(preFactor, preFactorGrad, gradThetaGamma, gradUpsilon1, gradUpsilon2, compUpsilon1, compUpsilon2, mode, term=None):

    # LFMGRADIENTH31 Gradient of the function h_i(z) with respect to some of the
    # hyperparameters of the kernel: m_k, C_k, D_k, m_r, C_r or D_r.
    # FORMAT
    # DESC Computes the gradient of the function h_i(z) with respect to some of
    # the parameters of the system (mass, spring or damper).
    # ARG gamma1 : Gamma value for first system.
    # ARG gamma2 : Gamma value for second system.
    # ARG sigma2 : length scale of latent process.
    # ARG gradThetaGamma : Vector with the gradient of gamma1 and gamma2 with
    # respect to the desired parameter.
    # ARG X : first time input (number of time points x 1).
    # ARG X2 : second time input (number of time points x 1)
    # ARG mode: indicates in which way the vectors X and X2 must be transposed
    # RETURN g : Gradient of the function with respect to the desired
    # parameter.
    #
    # Author : Tianqi Wei
    # Based on Matlab codes by David Luengo, 2007, 2008, Mauricio Alvarez, 2008
    print('preFactor')
    print(preFactor)
    print('preFactorGrad')
    print(preFactorGrad)
    if not mode:
        if not term:
            g = (preFactor*gradUpsilon1 + preFactorGrad*compUpsilon1)*gradThetaGamma
        else:
            g = (-preFactor[0]*gradUpsilon1 + preFactorGrad[0]*compUpsilon1)*gradThetaGamma[0] \
                +(preFactor[1]*np.conj(gradUpsilon1) - preFactorGrad[1]*np.conj(compUpsilon1))*gradThetaGamma[1]
    else:
        g = (preFactor[0]*gradUpsilon1 + preFactorGrad[0]*compUpsilon1)*gradThetaGamma[0] \
            + (preFactor[1]*gradUpsilon2 + preFactorGrad[1]*compUpsilon2)*gradThetaGamma[1]
    return g


def lfmGradientH32(preFactor, gradThetaGamma, compUpsilon1,compUpsilon2, mode, term=None):
    # LFMGRADIENTH32 Gradient of the function h_i(z) with respect to some of the
    # hyperparameters of the kernel: m_k, C_k, D_k, m_r, C_r or D_r.
    # FORMAT
    # DESC Computes the gradient of the function h_i(z) with respect to some of
    # the parameters of the system (mass, spring or damper).
    # ARG gamma1 : Gamma value for first system.
    # ARG gamma2 : Gamma value for second system.
    # ARG sigma2 : length scale of latent process.
    # ARG gradThetaGamma : Vector with the gradient of gamma1 and gamma2 with
    # respect to the desired parameter.
    # ARG X : first time input (number of time points x 1).
    # ARG X2 : second time input (number of time points x 1)
    # ARG mode: indicates in which way the vectors X and X2 must be transposed
    # RETURN g : Gradient of the function with respect to the desired
    # parameter.
    #
    # Author : Tianqi Wei
    # Based on Matlab codes by David Luengo, 2007, 2008, Mauricio Alvarez, 2008

    if not mode:
        if not term:
            g = compUpsilon1*(-(gradThetaGamma[1]/preFactor[1]) + (gradThetaGamma[0]/preFactor[0]))
        else:
            g = (compUpsilon1*preFactor[0] - np.conj(compUpsilon1)*preFactor[1])*gradThetaGamma
    else:
        g = compUpsilon1*(-(gradThetaGamma[1]/preFactor[2]) + (gradThetaGamma[0]/preFactor[0])) \
            + compUpsilon2*(-(gradThetaGamma[0]/preFactor[1]) + (gradThetaGamma[1]/preFactor[3]))
    return g


def lfmGradientH41(preFactor, preFactorGrad, gradThetaGamma, preExp, gradUpsilon1, gradUpsilon2, compUpsilon1, compUpsilon2, mode, term=None):
    # LFMGRADIENTH41 Gradient of the function h_i(z) with respect to some of the
    # hyperparameters of the kernel: m_k, C_k, D_k, m_r, C_r or D_r.
    # FORMAT
    # DESC Computes the gradient of the function h_i(z) with respect to some of
    # the parameters of the system (mass, spring or damper).
    # ARG gamma1 : Gamma value for first system.
    # ARG gamma2 : Gamma value for second system.
    # ARG sigma2 : length scale of latent process.
    # ARG gradThetaGamma : Vector with the gradient of gamma1 and gamma2 with
    # respect to the desired parameter.
    # ARG X : first time input (number of time points x 1).
    # ARG X2 : second time input (number of time points x 1)
    # ARG mode: indicates in which way the vectors X and X2 must be transposed
    # RETURN g : Gradient of the function with respect to the desired
    # parameter.
    #
    # Author : Tianqi Wei
    # Based on Matlab codes by David Luengo, 2007, 2008, Mauricio Alvarez, 2008

    if not mode:
        if not term:
            g = (gradUpsilon1 * gradThetaGamma) * (preExp/preFactor[0] - np.conj(preExp)/preFactor[1]).T \
                + (compUpsilon1 * gradThetaGamma) * (- preExp/preFactorGrad[0] + np.conj(preExp)/preFactorGrad[1]).T
        else:
            g = (gradUpsilon1 * gradThetaGamma[0]) * (preExp/preFactor[0]).T \
                + (compUpsilon1 * gradThetaGamma[0]) * (- preExp/preFactorGrad[0]).T \
                + (np.conj(gradUpsilon1) * gradThetaGamma[1]) * (- preExp/preFactor[1]).T \
                + (np.conj(compUpsilon1) * gradThetaGamma[1]) * (preExp/preFactorGrad[1]).T
    else:
        g = (gradUpsilon1 * gradThetaGamma[0]) * (preExp[:, 0]/preFactor[0] - preExp[:, 1]/preFactor[1]).T \
            + (compUpsilon1 * gradThetaGamma[0]) * (- preExp[:, 0]/preFactorGrad[0] + preExp[:, 1]/preFactorGrad[1]).T \
            + (gradUpsilon2 * gradThetaGamma[1]) * (preExp[:, 1]/preFactor[3] - preExp[:, 0]/preFactor[2]).T \
            + (compUpsilon2 * gradThetaGamma[1]) * (- preExp[:, 1]/preFactorGrad[3] + preExp[:, 0]/preFactorGrad[2]).T
    return g


def lfmGradientH42(preFactor, preFactorGrad, gradThetaGamma, preExp, preExpt,
    compUpsilon1, compUpsilon2, mode, term=None):
    # LFMGRADIENTH42 Gradient of the function h_i(z) with respect to some of the
    # hyperparameters of the kernel: m_k, C_k, D_k, m_r, C_r or D_r.
    # FORMAT
    # DESC Computes the gradient of the function h_i(z) with respect to some of
    # the parameters of the system (mass, spring or damper).
    # ARG gamma1 : Gamma value for first system.
    # ARG gamma2 : Gamma value for second system.
    # ARG sigma2 : length scale of latent process.
    # ARG gradThetaGamma : Vector with the gradient of gamma1 and gamma2 with
    # respect to the desired parameter.
    # ARG X : first time input (number of time points x 1).
    # ARG X2 : second time input (number of time points x 1)
    # ARG mode: indicates in which way the vectors X and X2 must be transposed
    # RETURN g : Gradient of the function with respect to the desired
    # parameter.
    #
    # Author : Tianqi Wei
    # Based on Matlab codes by David Luengo, 2007, 2008, Mauricio Alvarez, 2008

    if not mode:
        if not term:
            g = compUpsilon1*(- (preExp/preFactorGrad[0] + preExpt/preFactor[0])*gradThetaGamma[0]
                + (np.conj(preExp)/preFactorGrad[1] + np.conj(preExpt)/preFactor[1])*gradThetaGamma[1]).T
        else:
            g = compUpsilon1*(- (preExp/preFactorGrad[0] + preExpt/preFactor[0])*gradThetaGamma).T \
                + np.conj(compUpsilon1)*((preExp/preFactorGrad[1] + preExpt/preFactor[1])*gradThetaGamma).T
    else:
        g = compUpsilon1*((preExp[:, 1]/preFactorGrad[2] + preExpt[:, 1]/preFactor[2])*gradThetaGamma[1]
            - (preExp[:, 0]/preFactorGrad[0] + preExpt[:, 0]/preFactor[0])*gradThetaGamma[0]).T \
            - compUpsilon2*((preExp[:, 1]/preFactorGrad[3] + preExpt[:, 1]/preFactor[3])*gradThetaGamma[1]
            - (preExp[:, 0]/preFactorGrad[1] + preExpt[:, 0]/preFactor[1])*gradThetaGamma[0]).T
    return g


def lfmGradientSigmaH3(gamma1, gamma2, sigma2, X, X2, preFactor, mode, term=None):

    # LFMGRADIENTSIGMAH3 Gradient of the function h_i(z) with respect \sigma.
    # FORMAT
    # DESC Computes the gradient of the function h_i(z) with respect to the
    # length-scale of the input "force", \sigma.
    # ARG gamma1 : Gamma value for first system.
    # ARG gamma2 : Gamma value for second system.
    # ARG sigma2 : length scale of latent process.
    # ARG X : first time input (number of time points x 1).
    # ARG X2 : second time input (number of time points x 1).
    # ARG mode: indicates in which way the vectors X and X2 must be transposed
    # RETURN g : Gradient of the function with respect to \sigma.
    #
    # Author : Tianqi Wei
    # Based on Matlab codes by David Luengo, 2007, 2008, Mauricio Alvarez, 2008


    if not mode:
        if not term:
            g = preFactor * lfmGradientSigmaUpsilonMatrix(gamma1, sigma2, X, X2)
        else:
            gradupsilon = lfmGradientSigmaUpsilonMatrix(gamma1, sigma2, X, X2)
            g = -preFactor[0] * gradupsilon + preFactor[1] * np.conj(gradupsilon)
    else:
        g = preFactor[0] * lfmGradientSigmaUpsilonMatrix(gamma1, sigma2, X, X2) + \
            preFactor[1] * lfmGradientSigmaUpsilonMatrix(gamma2, sigma2, X, X2)
    return g


def lfmGradientSigmaH4(gamma1, gamma2, sigma2, X, preFactor, preExp, mode, term):
    # LFMGRADIENTSIGMAH4 Gradient of the function h_i(z) with respect \sigma.
    # FORMAT
    # DESC Computes the gradient of the function h_i(z) with respect to the
    # length-scale of the input "force", \sigma.
    # ARG gamma1 : Gamma value for first system.
    # ARG gamma2 : Gamma value for second system.
    # ARG sigma2 : length scale of latent process.
    # ARG X : first time input (number of time points x 1).
    # ARG X2 : second time input (number of time points x 1).
    # ARG mode: indicates in which way the vectors X and X2 must be transposed
    # RETURN g : Gradient of the function with respect to \sigma.
    #
    # Author : Tianqi Wei
    # Based on Matlab codes by David Luengo, 2007, 2008, Mauricio Alvarez, 2008
    if not mode:
        if not term:
            g = lfmGradientSigmaUpsilonVector(gamma1, sigma2, X)*(preExp/preFactor[0] - np.conj(preExp)/preFactor[1]).T
        else:
            gradupsilon = lfmGradientSigmaUpsilonVector(gamma1, sigma2, X)
            g = gradupsilon * (preExp/preFactor[0]).T - np.conj(gradupsilon)*(preExp/preFactor[1]).T
    else:
        g = lfmGradientSigmaUpsilonVector(gamma1, sigma2, X)*(preExp[:, 0]/preFactor[0] - preExp[:, 1]/preFactor[1]).T \
            + lfmGradientSigmaUpsilonVector(gamma2, sigma2, X)*(preExp[:, 1]/preFactor[2] - preExp[:, 0]/preFactor[3]).T
    return g

    
def lfmDiagComputeH3(gamma, sigma2, t, factor, preExp, mode):
    upsi = lfmUpsilonVector(gamma ,sigma2, t)
    if mode:
        vec = preExp*upsi*factor
    else:
        temp = preExp*upsi
        vec = 2*np.real(temp/factor[0]) - temp/factor[1]
    return [vec, upsi] 


def lfmDiagComputeH4(gamma, sigma2, t, factor, preExp, mode):
    upsi = lfmUpsilonVector(gamma ,sigma2, t)
    if mode:
        vec = (preExp[:,0]/factor[0] -  2*preExp[:,1]/factor[1])*upsi
    else:
        temp2 = upsi*np.conj(preExp)/factor[1]
        vec = upsi*preExp/factor[0] - 2*np.real(temp2)
    return [vec, upsi]


def lfmDiagGradientH3(gamma, t, factor, preExp, compUpsilon, gradUpsilon, termH, preFactorGrad, gradTheta):
    expUpsilon = preExp*compUpsilon
    pgrad = - expUpsilon*(2/preFactorGrad**2)*gradTheta[0] - \
            (t*termH + preExp*gradUpsilon*factor[0] - expUpsilon*(1/gamma**2 - 2/preFactorGrad**2))*gradTheta[1]
    return pgrad


def lfmDiagGradientH4( t, factor, preExp, compUpsilon, gradUpsilon, gradTheta):
    pgrad = 2*preExp[:,1] * compUpsilon * (t/factor[1] + 1/factor[1]**2)*gradTheta[0] \
            + (gradUpsilon * (preExp[:,0]/factor[0] - 2*preExp[:,1]/factor[1])
            - compUpsilon * (t * preExp[:,0]/factor[0] + preExp[:,0]/factor[0]**2
            -  2*preExp[:,1]/factor[1]**2))*gradTheta[1]
    return pgrad


def lfmDiagGradientSH3(gamma, sigma2, t, factor, preExp, mode):
    upsi = lfmGradientSigmaUpsilonVector(gamma ,sigma2, t)
    if mode:
        vec = preExp * upsi*factor
    else:
        temp = preExp * upsi
        vec = 2*np.real(temp/factor[0]) - temp/factor[1]
    return [vec, upsi]


def lfmDiagGradientSH4(gamma, sigma2, t, factor, preExp, mode):
    upsi = lfmGradientSigmaUpsilonVector(gamma ,sigma2, t)
    if mode:
        vec = (preExp[:, 0]/factor[0] - 2*preExp[:, 1]/factor[1]) * upsi
    else:
        temp2 = upsi * np.conj(preExp)/factor[1]
        vec = upsi * preExp/factor[0] - 2*np.real(temp2)
    return [vec, upsi]


class UnilateralKernelParameters:
    def __init__(self, type="LFM", name="LFM"):
        self.type = "LFM"
        self.name = "LFM"


class LFM(Kern):
    """
    Covariance function for intrinsic/linear coregionalization models

    This covariance has the form:
    .. math::
       \mathbf{B} = \mathbf{W}\mathbf{W}^\top + \text{diag}(kappa)

    An intrinsic/linear coregionalization covariance function of the form:
    .. math::

       k_2(x, y)=\mathbf{B} k(x, y)

    it is obtained as the tensor product between a covariance function
    k(x, y) and B.

    :param output_dim: number of outputs to coregionalize
    :type output_dim: int
    :param rank: number of columns of the W matrix (this parameter is ignored if parameter W is not None)
    :type rank: int
    :param W: a low rank matrix that determines the correlations between the different outputs, together with kappa it forms the coregionalization matrix B
    :type W: numpy array of dimensionality (num_outpus, W_columns)
    :param kappa: a vector which allows the outputs to behave independently
    :type kappa: numpy array of dimensionality  (output_dim, )

    .. note: see coregionalization examples in GPy.examples.regression for some usage.
    """


    def __init__(self, unilateral_kernel_types, input_dim=2, active_dims=None, inv_lengthscales=None, mass=None,
                 spring=None, damper=None, sensitivity=None,  isNormalised=None, name='multigp'):
        super(LFM, self).__init__(input_dim, active_dims, name)
        # assert len(unilateral_kernel_types) == len(input_sizes), "the input_types should has the same length with input_size"
        self.kernel_types = unilateral_kernel_types
        # self.input_sizes = input_sizes
        # self.input_dims = input_dims
        self.type_length = len(unilateral_kernel_types)
        # self.inv_lengthscales =  Param('inv_lengthscales', inv_lengthscales)
        # self.link_parameter(self.inv_lengthscales)
        self.output_dim = len(unilateral_kernel_types)

        self.unilateral_kernels = cell(self.type_length, 1)
        for i in range(self.type_length):
            if self.kernel_types[i] == "RBF":
                self.unilateral_kernels[i] = UnilateralKernelParameters(type="RBF", name="RBF")
                self.unilateral_kernels[i].inv_l = Param('inv_lengthscale', 1./self.lengthscale[i]**2, Logexp())
                self.link_parameter(self.inv_l)
                if isNormalised is None:
                    isNormalised = True
                self.unilateral_kernels[i].isNormalised = next(isNormalised)

            if self.kernel_types[i] == "LFM":
                self.unilateral_kernels[i] = UnilateralKernelParameters(type="LFM", name="LFM")
                self.unilateral_kernels[i].inv_l = Param('inv_lengthscale', inv_lengthscales[i], Logexp())
                if mass is None:
                    mass = 1
                self.unilateral_kernels[i].mass = Param('scale', next(mass))
                if spring is None:
                    spring = 1
                self.unilateral_kernels[i].spring = Param('scale', next(spring))
                if damper is None:
                    damper = 1
                self.unilateral_kernels[i].damper = Param('scale', next(damper))
                if sensitivity is None:
                    sensitivity = 1
                self.unilateral_kernels[i].sensitivity = Param('sensitivity', next(sensitivity))

                self.link_parameters(self.unilateral_kernels[i].inv_l,
                                     self.unilateral_kernels[i].mass,
                                     self.unilateral_kernels[i].spring,
                                     self.unilateral_kernels[i].damper,
                                     self.unilateral_kernels[i].sensitivity)
                if isNormalised is None:
                    isNormalised = True
                self.unilateral_kernels[i].isNormalised = next(isNormalised)

        self.cross_kernel_matrix = cell(self.type_length, self.type_length)
        for i in range(self.type_length):
            for j in range(self.type_length):
                self.cross_kernel_matrix[i][j] = unilateral_kernel_types[i] + "X" + unilateral_kernel_types[j]

        self.recalculate_intermediate_variables()

        # The kernel ALLWAYS puts the output index (the q for qth output) in the end of each rows.
        self.index_dim = -1


    def recalculate_intermediate_variables(self):
        for i in range(self.type_length):
            if self.kernel_types[i]  == "LFM":
                self.unilateral_kernels[i].sigma2 = 2/self.unilateral_kernels[i].inv_l
                self.unilateral_kernels[i].sigma = csqrt(self.unilateral_kernels[i].sigma2)
                # alpha and omega are intermediate variables used in the model and gradient for optimisation
                self.unilateral_kernels[i].alpha = self.unilateral_kernels[i].damper / (2 * self.unilateral_kernels[i].mass)
                self.unilateral_kernels[i].omega = csqrt(self.unilateral_kernels[i].spring / self.unilateral_kernels[i].mass - self.unilateral_kernels[i].alpha * self.unilateral_kernels[i].alpha)
                self.unilateral_kernels[i].omega_isreal = np.isreal(self.unilateral_kernels[i].omega)
                self.unilateral_kernels[i].gamma = self.unilateral_kernels[i].alpha + 1j * self.unilateral_kernels[i].omega


    def parameters_changed(self):
        '''
        This function overrides the same name function in the grandparent class "Parameterizable", which is simply
        "pass"
        It describes the behaviours of the class when the "parameters" of a kernel are updated.
        '''
        self.recalculate_intermediate_variables()
        super(LFM, self).parameters_changed()


    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        slices = index_to_slices(X[:, self.index_dim])
        slices2 = index_to_slices(X2[:, self.index_dim])
        target = np.zeros((X.shape[0], X2.shape[0]))
        for i in range(len(slices)):
            for j in range(len(slices2)):
                for k in range(len(slices[i])):
                    for l in range(len(slices2[j])):
                        K_sub_matrix = self.K_sub_matrix(i, X[slices[i][k], :-1], j, X2[slices2[j][l], :-1])
                        target.__setitem__((slices[i][k], slices2[j][l]), K_sub_matrix)
        return target

    def K_sub_matrix(self, q1, X, q2=None, X2=None):
        if X2 is None:
            X2 = X
        if q2 is None:
            q2 = q1
        if self.unilateral_kernels[q1].type == "LFM" and self.unilateral_kernels[q2].type == "LFM":
            K_sub = self.K_sub_matrix_LFMXLFM(self.unilateral_kernels[q1], X, self.unilateral_kernels[q2], X2)
        elif self.unilateral_kernels[q1].type == "LFM" and self.unilateral_kernels[q2].type == "RBF":
            K_sub = self.K_sub_matrix_LFMXRBF(self.unilateral_kernels[q1], X, self.unilateral_kernels[q2], X2)
        elif self.unilateral_kernels[q1].type == "RBF" and self.unilateral_kernels[q2].type == "RBF":
            K_sub = self.K_sub_matrix_RBFXRBF(self.unilateral_kernels[q1], X, self.unilateral_kernels[q2], X2)
        return K_sub

    def K_sub_matrix_LFMXLFM(self, q1, X, q2= None, X2=None):
        # This K is a sub matrix as a part of K. It is covariance matrix between a pair of outputs.
        # The variable q and q2 are the index of outputs.
        # The variable X and X2 are subset of the X and X2 in K.
        if X2 is None:
            X2 = X
        if q2 is None:
            q2 = q1
        assert (X.shape[1] == 1 and X.shape[1] == 1), 'Input can only have one column'# + str(X) + str(X2)

        # Creation  of the time matrices

        if self.unilateral_kernels[q1].omega_isreal and self.unilateral_kernels[q2].omega_isreal :
            # Pre-computations to increase speed
            gamma1 = self.unilateral_kernels[q1].alpha + 1j * self.unilateral_kernels[q1].omega
            gamma2 = self.unilateral_kernels[q2].alpha + 1j * self.unilateral_kernels[q2].omega
            # print('gamma1')
            # print(gamma1)
            # print('gamma2')
            # print(gamma2)
            preGamma = np.array([gamma1 + gamma2,
                                 np.conj(gamma1) + gamma2
                                ])
            preConsX = 1. / preGamma
            preExp1 = np.exp(-gamma1 * X)
            preExp2 = np.exp(-gamma2 * X2)
            # Actual computation  of the kernel
            sK = np.real(
                        lfmComputeH3(gamma1, gamma2, self.unilateral_kernels[q1].scale, X, X2, preConsX, 0, 1)[0]
                        + lfmComputeH3(gamma2, gamma1, self.unilateral_kernels[q1].scale, X2, X, preConsX[1] - preConsX[0], 0, 0)[0].T
                        + lfmComputeH4(gamma1, gamma2, self.unilateral_kernels[q1].scale, X, preGamma, preExp2, 0, 1)[0]
                        + lfmComputeH4(gamma2, gamma1, self.unilateral_kernels[q1].scale, X2, preGamma, preExp1, 0, 0)[0].T
                        )
            if self.unilateral_kernels[q1].isNormalised:
                K0 = (self.unilateral_kernels[q1].sensitivity * self.sensitivity[q2]) / (
                        4 * csqrt(2) * self.unilateral_kernels[q1].mass * self.unilateral_kernels[q2].mass
                        * self.unilateral_kernels[q1].omega*self.unilateral_kernels[q2].omega)
            else:
                K0 = (csqrt(self.unilateral_kernels[q1].sigma) * csqrt(np.pi) * self.unilateral_kernels[q1].sensitivity * self.unilateral_kernels[q2].sensitivity) / (
                        4 * self.unilateral_kernels[q1].mass * self.unilateral_kernels[q2].mass * self.unilateral_kernels[q1].omega*self.unilateral_kernels[q2].omega)

            K = K0 * sK
        else:
            # Pre-computations to increase the speed
            preExp1 = np.zeros((np.max(np.shape(X)), 2))
            preExp2 = np.zeros((np.max(np.shape(X2)), 2))
            gamma1_p = self.unilateral_kernels[q1].alpha + 1j * self.unilateral_kernels[q1].omega
            gamma1_m = self.unilateral_kernels[q1].alpha - 1j * self.unilateral_kernels[q1].omega
            gamma2_p = self.unilateral_kernels[q2].alpha + 1j * self.unilateral_kernels[q2].omega
            gamma2_m = self.unilateral_kernels[q2].alpha - 1j * self.unilateral_kernels[q2].omega
            preGamma = np.array([   gamma1_p + gamma2_p,
                                    gamma1_p + gamma2_m,
                                    gamma1_m + gamma2_p,
                                    gamma1_m + gamma2_m
                                ])
            preConsX = 1. / preGamma
            preFactors = np.array([ preConsX[1] - preConsX[0],
                                    preConsX[2] - preConsX[3],
                                    preConsX[2] - preConsX[0],
                                    preConsX[1] - preConsX[3]
                                ])
            preExp1[:, 0] = np.exp(-gamma1_p * X).ravel()
            preExp1[:, 1] = np.exp(-gamma1_m * X).ravel()
            preExp2[:, 0] = np.exp(-gamma2_p * X2).ravel()
            preExp2[:, 1] = np.exp(-gamma2_m * X2).ravel()
            # Actual  computation of the kernel
            sK = (
                    lfmComputeH3(gamma1_p, gamma1_m, self.unilateral_kernels[q1].scale, X, X2, preFactors[np.array([0, 1])], 1)[0]
                    + lfmComputeH3(gamma2_p, gamma2_m, self.unilateral_kernels[q1].scale, X2, X, preFactors[np.array([2, 3])], 1)[0].T
                    + lfmComputeH4(gamma1_p, gamma1_m, self.unilateral_kernels[q1].scale, X, preGamma[np.array([0, 1, 3, 2])], preExp2, 1)[0]
                    + lfmComputeH4(gamma2_p, gamma2_m, self.unilateral_kernels[q1].scale, X2, preGamma[np.array([0, 2, 3, 1])], preExp1, 1)[0].T
                )
            if self.unilateral_kernels[q1].isNormalised:
                K0 = (self.unilateral_kernels[q1].sensitivity * self.unilateral_kernels[q2].sensitivity) \
                     / (8 * csqrt(2) * self.unilateral_kernels[q1].mass * self.unilateral_kernels[q2].mass *
                        self.unilateral_kernels[q1].omega * self.unilateral_kernels[q2].omega)
            else:
                K0 = (csqrt(self.unilateral_kernels[q1].scale) * csqrt(np.pi) *
                      self.unilateral_kernels[q1].sensitivity * self.unilateral_kernels[q2].sensitivity) \
                     / (8 * self.unilateral_kernels[q1].mass * self.unilateral_kernels[q2].mass
                        * self.unilateral_kernels[q1].omega * self.unilateral_kernels[q2].omega)
            K = K0 * sK
        return K


    def Kdiag(self, X):
        assert X.shape[1] == 2, 'Input can only have one column'
        slices = index_to_slices(X[:,self.index_dim])
        target = np.zeros((X.shape[0]))  #.astype(np.complex128)
        for q1, slices_i in zip(range(self.output_dim), slices):
            for s in slices_i:
                Kdiag_sub = np.real(self.Kdiag_sub(q1, X[s, :-1]))
                np.copyto(target[s], Kdiag_sub)
        return target


    def Kdiag_sub_matrix(self, q1, X):
        if self.unilateral_kernels[q1].type == "LFM":
            Kdiag_sub = self.Kdiag_sub_LFM(q1, X)
        elif self.unilateral_kernels[q1].type == "RBF":
            Kdiag_sub = self.Kdiag_sub_RBF(q1, X)
        return Kdiag_sub


    def Kdiag_sub_LFM(self, q1, X):

        def lfmDiagComputeH3(gamma, sigma2, t, factor, preExp, mode):
            if mode:
                vec = np.multiply(preExp, lfmUpsilonVector(gamma, sigma2, t)) * factor
            else:
                temp = np.multiply(preExp, lfmUpsilonVector(gamma, sigma2, t))
                vec = 2 * np.real(temp / factor[0]) - temp / factor[1]
            return vec

        def lfmDiagComputeH4(gamma, sigma2, t, factor, preExp, mode):
            if mode:
                vec = (preExp[:, 0] / factor[0] - 2 * preExp[:, 1] / factor[1]) * lfmUpsilonVector(gamma, sigma2, t)
            else:
                temp = lfmUpsilonVector(gamma, sigma2, t)
                temp2 =temp * np.conj(preExp) / factor[1]
                vec = (temp*preExp) / factor[0] - 2 * np.real(temp2)
            return vec

       # preExp = np.zeros((len(X), 2)).astype(np.complex128)
        gamma_p = self.unilateral_kernels[q1].alpha + 1j * self.unilateral_kernels[q1].omega
        gamma_m = self.unilateral_kernels[q1].alpha - 1j * self.unilateral_kernels[q1].omega
        preFactors = np.array([2 / (gamma_p + gamma_m) - 1 / gamma_m,
                               2 / (gamma_p + gamma_m) - 1 / gamma_p])
        preExp = np.hstack([np.exp(-gamma_p * X), np.exp(-gamma_m * X)])
        sigma2 = self.unilateral_kernels[q1].scale
        # Actual computation of the kernel
        sk = lfmDiagComputeH3(-gamma_m, sigma2, X, preFactors[0], preExp[:, 1], 1)  \
            + lfmDiagComputeH3(-gamma_p, sigma2, X, preFactors[1], preExp[:, 0], 1)  \
            + lfmDiagComputeH4(gamma_m, sigma2, X, [gamma_m, (gamma_p + gamma_m)], np.hstack([preExp[:, 1][:,None], preExp[:, 0][:,None]]), 1)  \
            + lfmDiagComputeH4(gamma_p, sigma2, X, [gamma_p, (gamma_p + gamma_m)], preExp, 1)
        if self.isNormalised:
            k0 = self.unilateral_kernels[q1].sensitivity ** 2 / (8 * csqrt(2) * self.unilateral_kernels[q1].mass ** 2 * self.unilateral_kernels[q1].omega ** 2)
        else:
            k0 = csqrt(np.pi) * self.unilateral_kernels[q1].sigma * self.unilateral_kernels[q1].sensitivity ** 2 / (8 * self.unilateral_kernels[q1].mass ** 2 * self.unilateral_kernels[q1].omega ** 2)
        k = k0 * sk
        return k


    def reset_gradients(self):
        self.scale.gradient = np.zeros_like(self.scale.gradient)
        self.mass.gradient = np.zeros_like(self.mass.gradient)
        self.spring.gradient = np.zeros_like(self.spring.gradient)
        self.damper.gradient = np.zeros_like(self.damper.gradient)
        self.sensitivity.gradient = np.zeros_like(self.sensitivity.gradient)

    def update_gradients_full(self, dL_dK, X, X2=None):
        self.reset_gradients()
        if X2 is None:
            X2 = X
        slices = index_to_slices(X[:, self.index_dim])
        slices2 = index_to_slices(X2[:, self.index_dim])
        normaliseRegardingToBatchSize = 0
        g = np.zeros((self.output_dim, 5))
        slices2 = index_to_slices(X2[:, self.index_dim])
        for j in range(len(slices2)):
            for i in range(len(slices)):
                for l in range(len(slices2[j])):
                    for k in range(len(slices[i])):
                        if self.unilateral_kernels[i].type == "LFM" and self.unilateral_kernels[i].type == "LFM":
                            [g1,g2]=self._update_gradients_LFMXLFM(i, j, X[slices[i][k], :-1], X2[slices2[j][l], :-1],
                                                           dL_dK[slices[i][k], slices2[j][l]])
                        elif self.unilateral_kernels[i].type == "LFM" and self.unilateral_kernels[i].type == "RBF":
                            [g1,g2]=self._update_gradients_LFMXRBF(i, j, X[slices[i][k], :-1], X2[slices2[j][l], :-1],
                                                           dL_dK[slices[i][k], slices2[j][l]])
                        elif self.unilateral_kernels[i].type == "RBF" and self.unilateral_kernels[i].type == "RBF":
                            [g1,g2]=self._update_gradients_RBFXRBF(i, j, X[slices[i][k], :-1], X2[slices2[j][l], :-1],
                                                           dL_dK[slices[i][k], slices2[j][l]])
                        normaliseRegardingToBatchSize +=1
                        g[i] += g1 + g2
                        #g[j]
        normalisedg = g.sum(axis=0)/normaliseRegardingToBatchSize
        self.scale.gradient += (normalisedg[:, 3])
        self.mass.gradient += normalisedg[:, 0]
        self.spring.gradient += normalisedg[:, 1]
        self.damper.gradient += normalisedg[:, 2]
        self.sensitivity.gradient += normalisedg[:, 4]

    def _update_gradients_LFMXLFM(self, q1, q2, X, X2=None, dL_dK=None, meanVector=None):
        """
        Given the derivative of the objective wrt the covariance matrix
        (dL_dK), compute the gradient wrt the parameters of this kernel,
        and store in the parameters object as e.g. self.variance.gradient
        """
        # FORMAT
        # DESC computes cross kernel terms between two LFM kernels for
        # the multiple output kernel.
        # ARG lfmKern1 : the kernel structure associated with the first LFM
        # kernel.
        # ARG lfmKern2 : the kernel structure associated with the second LFM
        # kernel.
        # ARG X : row inputs for which kernel is to be computed.
        # ARG t2 : column inputs for which kernel is to be computed.
        # ARG covGrad : gradient of the objective function with respect to
        # the elements of the cross kernel matrix.
        # ARG meanVec : precomputed factor that is used for the switching dynamical
        # latent force model.
        # RETURN g1 : gradient of the parameters of the first kernel, for
        # ordering see lfmKernExtractParam.
        # RETURN g2 : gradient of the parameters of the second kernel, for
        # ordering see lfmKernExtractParam.
        #
        # SEEALSO : multiKernParamInit, multiKernCompute, lfmKernParamInit, lfmKernExtractParam
        #
        # COPYRIGHT : Tianqi Wei

        # Modified based on the Matlab codes by: David Luengo, 2007, 2008, Mauricio Alvarez, 2008
        #
        # MODIFICATIONS : Neil D. Lawrence, 2007
        #
        # MODIFICATIONS : Mauricio A. Alvarez, 2010

        # KERN

        subComponent = False  # This is just a flag that indicates if this kernel is part of a bigger kernel (SDLFM)
        covGrad = dL_dK
        if covGrad is None and meanVector is None:
            covGrad = X2
            X2 = X
        elif covGrad is not None and meanVector is not None:
            subComponent = True
            if np.size(meanVector) > 1:
                if np.shape(meanVector, 1) == 1:
                    assert np.shape(meanVector, 2) == np.shape(covGrad,
                                                               2), 'The dimensions of meanVector don''t correspond to the dimensions of covGrad'
                else:
                    assert np.shape(meanVector.conj().T, 2) == np.shape(covGrad,
                                                                        2), 'The dimensions of meanVector don''t correspond to the dimensions of covGrad'
            else:
                if np.size(X) == 1 and np.size(X2) > 1:
                    # matGrad will be row vector and so should be covGrad
                    dimcovGrad = np.max(np.shape((covGrad)))
                    covGrad = covGrad.reshape(1, dimcovGrad, order='F').copy()
                elif np.size(X) > 1 and np.size(X2) == 1:
                    # matGrad will be column vector and sp should be covGrad
                    dimcovGrad = np.shape.max(covGrad)
                    covGrad = covGrad.reshape(dimcovGrad, 1, order='F').copy()

        assert np.shape(X)[1] == 1 or np.shape(X2)[1] == 1, 'Input can only have one column. np.shape(X) = ' + str(
            np.shape(X)) + 'np.shape(X2) =' + str(np.shape(X2))
        assert self.unilateral_kernels[q1].scale == self.unilateral_kernels[
            q2].scale, 'Kernels cannot be cross combined if they have different inverse widths.'

        # Parameters of the simulation (in the order provided by kernExtractParam in the matlab code)
        index = np.array([q, q2])
        m = self.mass[index]  # Par. 1
        D = self.spring[index]  # Par. 2
        C = self.damper[index]  # Par. 3
        sigma2 = 2 / self.unilateral_kernels[q1].inv_l  # Par. 4
        sigma = csqrt(sigma2)
        S = self.sensitivity[index]  # Par. 5

        alpha = C / (2 * m)
        omega = csqrt(D / m - alpha ** 2)

        # Initialization of vectors and matrices

        g1 = np.zeros((5))
        g2 = np.zeros((5))

        # Precomputations

        if np.all(np.isreal(omega)):
            computeH = cell(4, 1)
            computeUpsilonMatrix = cell(2, 1)
            computeUpsilonVector = cell(2, 1)
            gradientUpsilonMatrix = cell(2, 1)
            gradientUpsilonVector = cell(2, 1)
            gamma1 = alpha[0] + 1j * omega[0]
            gamma2 = alpha[1] + 1j * omega[1]
            gradientUpsilonMatrix[0] = lfmGradientUpsilonMatrix(gamma1, sigma2, X, X2)
            gradientUpsilonMatrix[1] = lfmGradientUpsilonMatrix(gamma2, sigma2, X2, X)
            gradientUpsilonVector[0] = lfmGradientUpsilonVector(gamma1, sigma2, X)
            gradientUpsilonVector[1] = lfmGradientUpsilonVector(gamma2, sigma2, X2)
            preGamma = np.array([gamma1 + gamma2, np.conj(gamma1) + gamma2])
            preGamma2 = np.power(preGamma, 2)
            preConsX = 1 / preGamma
            preConsX2 = 1 / preGamma2
            preExp1 = np.exp(-gamma1 * X)
            preExp2 = np.exp(-gamma2 * X2)
            preExpX = np.multiply(X, np.exp(-gamma1 * X))
            preExpX2 = np.multiply(X2, np.exp(-gamma2 * X2))
            [computeH[0], computeUpsilonMatrix[0]] = lfmComputeH3(gamma1, gamma2, sigma2, X, X2, preConsX, 0, 1)
            [computeH[1], computeUpsilonMatrix[1]] = lfmComputeH3(gamma2, gamma1, sigma2, X2, X,
                                                                  preConsX[1] - preConsX[0], 0, 0)
            [computeH[2], computeUpsilonVector[0]] = lfmComputeH4(gamma1, gamma2, sigma2, X, preGamma, preExp2, 0, 1)
            [computeH[3], computeUpsilonVector[1]] = lfmComputeH4(gamma2, gamma1, sigma2, X2, preGamma, preExp1, 0, 0)
            preKernel = np.real(computeH[0] + computeH[1].T + computeH[2] + computeH[3].T)
        else:
            computeH = cell(4, 1)
            computeUpsilonMatrix = cell(2, 1)
            computeUpsilonVector = cell(2, 1)
            gradientUpsilonMatrix = cell(4, 1)
            gradientUpsilonVector = cell(4, 1)
            gamma1_p = alpha[0] + 1j * omega[0]
            gamma1_m = alpha[0] - 1j * omega[0]
            gamma2_p = alpha[1] + 1j * omega[1]
            gamma2_m = alpha[1] - 1j * omega[1]
            gradientUpsilonMatrix[0] = lfmGradientUpsilonMatrix(gamma1_p, sigma2, X, X2)
            gradientUpsilonMatrix[1] = lfmGradientUpsilonMatrix(gamma1_m, sigma2, X, X2)
            gradientUpsilonMatrix[2] = lfmGradientUpsilonMatrix(gamma2_p, sigma2, X2, X)
            gradientUpsilonMatrix[3] = lfmGradientUpsilonMatrix(gamma2_m, sigma2, X2, X)
            gradientUpsilonVector[0] = lfmGradientUpsilonVector(gamma1_p, sigma2, X)
            gradientUpsilonVector[1] = lfmGradientUpsilonVector(gamma1_m, sigma2, X)
            gradientUpsilonVector[2] = lfmGradientUpsilonVector(gamma2_p, sigma2, X2)
            gradientUpsilonVector[3] = lfmGradientUpsilonVector(gamma2_m, sigma2, X2)
            preExp1 = np.zeros((np.max(np.shape(X), 2)))
            preExp2 = np.zeros((np.max(np.shape(X2), 2)))
            preExpX = np.zeros((np.max(np.shape(X), 2)))
            preExpX2 = np.zeros((np.max(np.shape(X2), 2)))
            preGamma = np.array([gamma1_p + gamma2_p,
                                 gamma1_p + gamma2_m,
                                 gamma1_m + gamma2_p,
                                 gamma1_m + gamma2_m])
            preGamma2 = np.power(preGamma, 2)
            preConsX = 1 / preGamma
            preConsX2 = 1 / preGamma2
            preFactors = np.array([preConsX[1] - preConsX[0],
                                   preConsX[2] - preConsX[3],
                                   preConsX[2] - preConsX[0],
                                   preConsX[1] - preConsX[3]])
            preFactors2 = np.array([-preConsX2[1] + preConsX2[0],
                                    -preConsX2[2] + preConsX2[3],
                                    -preConsX2[2] + preConsX2[0],
                                    -preConsX2[1] + preConsX2[3]])
            preExp1[:, 0] = np.exp(-gamma1_p * X)
            preExp1[:, 1] = np.exp(-gamma1_m * X)
            preExp2[:, 0] = np.exp(-gamma2_p * X2)
            preExp2[:, 1] = np.exp(-gamma2_m * X2)
            preExpX[:, 0] = X * np.exp(-gamma1_p * X)
            preExpX[:, 1] = X * np.exp(-gamma1_m * X)
            preExpX2[:, 0] = X2 * np.exp(-gamma2_p * X2)
            preExpX2[:, 1] = X2 * np.exp(-gamma2_m * X2)
            [computeH[0], computeUpsilonMatrix[0]] = lfmComputeH3(gamma1_p, gamma1_m, sigma2, X, X2, preFactors[1, 2],
                                                                  1)
            [computeH[1], computeUpsilonMatrix[1]] = lfmComputeH3(gamma2_p, gamma2_m, sigma2, X2, X, preFactors[3, 4],
                                                                  1)
            [computeH[2], computeUpsilonVector[0]] = lfmComputeH4(gamma1_p, gamma1_m, sigma2, X, preGamma[1, 2, 4, 3],
                                                                  preExp2, 1)
            [computeH[3], computeUpsilonVector[1]] = lfmComputeH4(gamma2_p, gamma2_m, sigma2, X2, preGamma[1, 3, 4, 2],
                                                                  preExp1, 1)
            preKernel = computeH[0] + computeH[1].T + computeH[2] + computeH[3].T

        if np.all(np.isreal(omega)):
            if self.unilateral_kernels[q1].isNormalised:
                K0 = np.prod(S) / (4 * csqrt(2) * np.prod(m) * np.prod(omega))
            else:
                K0 = sigma * np.prod(S) * csqrt(np.pi) / (4 * np.prod(m) * np.prod(omega))
        else:
            if self.isNormalised[q2]:
                K0 = (np.prod(S) / (8 * csqrt(2) * np.prod(m) * np.prod(omega)))
            else:
                K0 = (sigma * np.prod(S) * csqrt(np.pi) / (8 * np.prod(m) * np.prod(omega)))

        # Gradient with respect to m, D and C
        for ind_theta in np.arange(3):  # Parameter (m, D or C)
            for ind_par in np.arange(2):  # System (1 or 2)
                # Choosing the right gradients for m, omega, gamma1 and gamma2
                if ind_theta == 0:  # Gradient wrt m
                    gradThetaM = [1 - ind_par, ind_par]
                    gradThetaAlpha = -C / (2 * np.power(m, 2))
                    gradThetaOmega = (np.power(C, 2) - 2 * m * D) / (2 * (m ** 2) * csqrt(4 * m * D - C ** 2))
                if ind_theta == 1:  # Gradient wrt D
                    gradThetaM = np.zeros((2))
                    gradThetaAlpha = np.zeros((2))
                    gradThetaOmega = 1 / csqrt(4 * m * D - np.power(C, 2))
                if ind_theta == 2:  # Gradient wrt C
                    gradThetaM = np.zeros((2))
                    gradThetaAlpha = 1 / (2 * m)
                    gradThetaOmega = -C / (2 * m * csqrt(4 * m * D - C ** 2))

                gradThetaGamma1 = gradThetaAlpha + 1j * gradThetaOmega
                gradThetaGamma2 = gradThetaAlpha - 1j * gradThetaOmega

                # Gradient evaluation
                if np.all(np.isreal(omega)):
                    gradThetaGamma2 = gradThetaGamma1[1]
                    gradThetaGamma1 = [gradThetaGamma1[0], np.conj(gradThetaGamma1[0])]

                    #  gradThetaGamma1 =  gradThetaGamma11
                    if not ind_par:
                        matGrad = K0 * \
                                  np.real(lfmGradientH31(preConsX, preConsX2, gradThetaGamma1,
                                                         gradientUpsilonMatrix[0], 1, computeUpsilonMatrix[0], 1, 0, 1)
                                          + lfmGradientH32(preGamma2, gradThetaGamma1, computeUpsilonMatrix[1],
                                                           1, 0, 0).T
                                          + lfmGradientH41(preGamma, preGamma2, gradThetaGamma1, preExp2,
                                                           gradientUpsilonVector[0], 1, computeUpsilonVector[0], 1, 0,
                                                           1)
                                          + lfmGradientH42(preGamma, preGamma2, gradThetaGamma1, preExp1, preExpX,
                                                           computeUpsilonVector[1], 1, 0, 0).T \
                                          - (gradThetaM[ind_par] / m[ind_par]
                                             + gradThetaOmega[ind_par] / omega[ind_par]) * preKernel)
                    else:
                        matGrad = K0 * \
                                  np.real(lfmGradientH31((preConsX[1] - preConsX[0]), (-preConsX2[1] + preConsX2[0]),
                                                         gradThetaGamma2, gradientUpsilonMatrix[1], 1,
                                                         computeUpsilonMatrix[1], 1, 0, 0).T
                                          + lfmGradientH32(preConsX2, gradThetaGamma2, computeUpsilonMatrix[0], 1, 0, 1)
                                          + lfmGradientH41(preGamma, preGamma2, gradThetaGamma2, preExp1,
                                                           gradientUpsilonVector[1], 1, computeUpsilonVector[1], 1, 0,
                                                           0).T
                                          + lfmGradientH42(preGamma, preGamma2, gradThetaGamma2, preExp2, preExpX2,
                                                           computeUpsilonVector[0], 1, 0, 1) \
                                          - (gradThetaM[ind_par] / m[ind_par] + gradThetaOmega[ind_par] / omega[
                                      ind_par])
                                          * preKernel)

                else:

                    gradThetaGamma11 = np.array([gradThetaGamma1[0], gradThetaGamma2[0]])
                    gradThetaGamma2 = np.array([gradThetaGamma1[1], gradThetaGamma2[1]])
                    gradThetaGamma1 = gradThetaGamma11

                    if not ind_par:  # ind_par = k
                        matGrad = K0 * \
                                  (lfmGradientH31(preFactors[np.array([0, 1])], preFactors2[np.array([0, 1])],
                                                  gradThetaGamma1, gradientUpsilonMatrix[0],
                                                  # preFactors[0,1], preFactors2[0,1], gradThetaGamma1, gradientUpsilonMatrix[0],
                                                  gradientUpsilonMatrix[1], computeUpsilonMatrix[0][0],
                                                  computeUpsilonMatrix[0][1], 1)
                                   + lfmGradientH32(preGamma2, gradThetaGamma1, computeUpsilonMatrix[1][0],
                                                    computeUpsilonMatrix[1][1], 1).T
                                   + lfmGradientH41(preGamma, preGamma2, gradThetaGamma1, preExp2,
                                                    gradientUpsilonVector[0],
                                                    gradientUpsilonVector[1], computeUpsilonVector[0][0],
                                                    computeUpsilonVector[0][1], 1)
                                   + lfmGradientH42(preGamma, preGamma2, gradThetaGamma1, preExp1, preExpX,
                                                    computeUpsilonVector[1][0],
                                                    computeUpsilonVector[1][1], 1).T
                                   - (gradThetaM[ind_par] / m[ind_par] + gradThetaOmega[ind_par] / omega[
                                              ind_par]) * preKernel)

                    else:  # ind_par = r
                        matGrad = K0 * \
                                  (lfmGradientH31(preFactors[2, 3], preFactors2[2, 3], gradThetaGamma2,
                                                  gradientUpsilonMatrix[2], gradientUpsilonMatrix[3],
                                                  computeUpsilonMatrix[1][0], computeUpsilonMatrix[1][1], 1).T
                                   + lfmGradientH32(preGamma2([0, 2, 1, 3]), gradThetaGamma2,
                                                    computeUpsilonMatrix[0][0],
                                                    computeUpsilonMatrix[0][1], 1)
                                   + lfmGradientH41(preGamma[0, 2, 1, 3], preGamma2[0, 2, 1, 3], gradThetaGamma2,
                                                    preExp1,
                                                    gradientUpsilonVector[2], gradientUpsilonVector[3],
                                                    computeUpsilonVector[1][0], computeUpsilonVector[1][1], 1).T
                                   + lfmGradientH42(preGamma[0, 2, 1, 3], preGamma2[0, 2, 1, 3], gradThetaGamma2,
                                                    preExp2,
                                                    preExpX2, computeUpsilonVector[0][0], computeUpsilonVector[0][1], 1)
                                   - (gradThetaM[ind_par] / m[ind_par] + gradThetaOmega[ind_par] / omega[ind_par])
                                   * preKernel)

                if subComponent:
                    if np.shape(meanVector)[0] == 1:
                        matGrad = matGrad * meanVector
                    else:
                        matGrad = (meanVector * matGrad).T
                # Check the parameter to assign the derivative
                if ind_par == 0:
                    g1[ind_theta] = sum(sum(matGrad * covGrad))
                else:
                    g2[ind_theta] = sum(sum(matGrad * covGrad))

        # Gradients with respect to sigma

        if np.all(np.isreal(omega)):
            if self.unilateral_kernels[q1].isNormalised:
                matGrad = K0 * \
                          np.real(lfmGradientSigmaH3(gamma1, gamma2, sigma2, X, X2, preConsX, 0, 1) \
                                  + lfmGradientSigmaH3(gamma2, gamma1, sigma2, X2, X, preConsX[1] - preConsX[0], 0, 0).T \
                                  + lfmGradientSigmaH4(gamma1, gamma2, sigma2, X, preGamma, preExp2, 0, 1) \
                                  + lfmGradientSigmaH4(gamma2, gamma1, sigma2, X2, preGamma, preExp1, 0, 0).T)
            else:
                matGrad = (np.prod(S) * csqrt(np.pi) / (4 * np.prod(m) * np.prod(omega))) \
                          * np.real(preKernel
                                    + sigma
                                    * (lfmGradientSigmaH3(gamma1, gamma2, sigma2, X, X2, preConsX, 0, 1)
                                       + lfmGradientSigmaH3(gamma2, gamma1, sigma2, X2, X, preConsX[1] - preConsX[0], 0,
                                                            0).T
                                       + lfmGradientSigmaH4(gamma1, gamma2, sigma2, X, preGamma, preExp2, 0, 1)
                                       + lfmGradientSigmaH4(gamma2, gamma1, sigma2, X2, preGamma, preExp1, 0, 0).T))
        else:
            if self.unilateral_kernels[q1].isNormalised:
                matGrad = K0 * \
                          (lfmGradientSigmaH3(gamma1_p, gamma1_m, sigma2, X, X2, preFactors[0, 1], 1) \
                           + lfmGradientSigmaH3(gamma2_p, gamma2_m, sigma2, X2, X, preFactors[2, 3], 1).T \
                           + lfmGradientSigmaH4(gamma1_p, gamma1_m, sigma2, X, preGamma[0, 1, 3, 2], preExp2, 1) \
                           + lfmGradientSigmaH4(gamma2_p, gamma2_m, sigma2, X2, preGamma[0, 2, 3, 1], preExp1, 1).T)
            else:
                matGrad = (np.prod(S) * csqrt(np.pi) / (8 * np.prod(m) * np.prod(omega))) \
                          * (preKernel
                             + sigma
                             * (lfmGradientSigmaH3(gamma1_p, gamma1_m, sigma2, X, X2, preFactors[0, 1], 1) \
                                + lfmGradientSigmaH3(gamma2_p, gamma2_m, sigma2, X2, X, preFactors[2, 3], 1).T \
                                + lfmGradientSigmaH4(gamma1_p, gamma1_m, sigma2, X, preGamma[0, 1, 3, 2], preExp2, 1) \
                                + lfmGradientSigmaH4(gamma2_p, gamma2_m, sigma2, X2, preGamma[0, 2, 3, 1], preExp1,
                                                     1).T))

        if subComponent:
            if np.shape(meanVector)[0] == 1:
                matGrad = matGrad * meanVector
            else:
                matGrad = (meanVector * matGrad).T

        g1[3] = sum(sum(matGrad * covGrad)) * (-np.power(sigma, 3) / 4)
        g2[3] = g1[3]

        # Gradients with respect to S

        if np.all(np.isreal(omega)):
            if self.unilateral_kernels[q1].isNormalised:
                matGrad = (1 / (4 * csqrt(2) * np.prod(m) * np.prod(omega))) * np.real(preKernel)
            else:
                matGrad = (sigma * csqrt(np.pi) / (4 * np.prod(m) * np.prod(omega))) * np.real(preKernel)
        else:
            if self.isNormalised[q2]:
                matGrad = (1 / (8 * csqrt(2) * np.prod(m) * np.prod(omega))) * (preKernel)
            else:
                matGrad = (sigma * csqrt(np.pi) / (8 * np.prod(m) * np.prod(omega))) * (preKernel)

        if subComponent:
            if np.shape(meanVector)[0] == 1:
                matGrad = matGrad * meanVector
            else:
                matGrad = (meanVector * matGrad).T
        g1[4] = sum(sum(S[1] * matGrad * covGrad))
        g2[4] = sum(sum(S[0] * matGrad * covGrad))

        g2[3] = 0  # Otherwise is counted twice

        g1 = np.real(g1)
        g2 = np.real(g2)
        # names = {'mass', 'spring', 'damper', 'inverse width', 'sensitivity'}
        # scale = 2/inverse width

        return [g1, g2]


    def _update_gradients_diag_wrapper(self, q1, X, dL_dKdiag):
        #  LFMKERNDIAGGRADIENT Compute the gradient of the LFM kernel's diagonal wrt parameters.
        #  FORMAT
        #  DESC computes the gradient of functions of the diagonal of the
        #  single input motif kernel matrix with respect to the parameters of the kernel. The
        #  parameters' gradients are returned in the order given by the
        #  lfmKernExtractParam command.
        #  ARG lfmKern : the kernel structure for which the gradients are
        #  computed.
        #  ARG X : the input data for which the gradient is being computed.
        #  ARG factors : partial derivatives of the function of interest with
        #  respect to the diagonal elements of the kernel.
        #  RETURN g : gradients of the relevant function with respect to each
        #  of the parameters. Ordering should match the ordering given in
        #  lfmKernExtractParam.

        assert np.shape(X)[1] == 1, 'Input can only have one column'

        # Parameters of the simulation (in the order providen by kernExtractParam)
        m = self.unilateral_kernels[q1].mass # Par. 1
        D = self.unilateral_kernels[q1].spring# Par. 2
        C = self.unilateral_kernels[q1].damper # Par. 3
        sigma2 = self.unilateral_kernels[q1].inv_l  # Par. 4
        sigma = csqrt(sigma2)
        S = self.unilateral_kernels[q1].sensitivity # Par. 5

        alpha = C / (2 * m)
        omega = csqrt(D / m - alpha ** 2)

        # Initialization of vectors and matrices
        g = np.zeros(( 5)) #########################

        # Precomputations
        diagH = cell(1, 4)
        gradDiag = cell(1, 2)
        upsilonDiag = cell(1, 4)
        preExp = np.zeros((len(X), 2))
        gamma_p = alpha + 1j * omega
        gamma_m = alpha - 1j * omega
        preFactors = np.array([2 / (gamma_p + gamma_m) - 1 / gamma_m,
                               2 / (gamma_p + gamma_m) - 1 / gamma_p])

        preExp[:, 0] = np.exp(-gamma_p * X)
        preExp[:, 1] = np.exp(-gamma_m * X)
        # Actual computation of the kernel
        [diagH[0], upsilonDiag[1]] = lfmDiagComputeH3(-gamma_m, sigma2, X, preFactors[0], preExp[:, 1], 1)
        [diagH[1], upsilonDiag[0]] = lfmDiagComputeH3(-gamma_p, sigma2, X, preFactors[1], preExp[:, 0], 1)
        [diagH[2], upsilonDiag[3]] = lfmDiagComputeH4(gamma_m, sigma2, X, [gamma_m(gamma_p + gamma_m)],
                                                      [preExp[:, 1], preExp[:, 0]], 1)
        [diagH[3], upsilonDiag[2]] = lfmDiagComputeH4(gamma_p, sigma2, X, [gamma_p(gamma_p + gamma_m)], preExp, 1)

        gradDiag[0] = lfmGradientUpsilonVector(-gamma_p, sigma2, X)
        gradDiag[1] = lfmGradientUpsilonVector(-gamma_m, sigma2, X)
        gradDiag[2] = lfmGradientUpsilonVector(gamma_p, sigma2, X)
        gradDiag[3] = lfmGradientUpsilonVector(gamma_m, sigma2, X)

        preKernel = diagH[0] + diagH[1] + diagH[2] + diagH[3]

        if self.unilateral_kernels[q1].isNormalised:
            k0 = np.power(self.unilateral_kernels[q1].sensitivity, 2) / (8 * csqrt(2) * np.power(self.unilateral_kernels[q1].mass, 2) * np.power(omega,2))
        else:
            k0 = csqrt(np.pi) * sigma * np.power(self.unilateral_kernels[q1].sensitivity, 2) / (8 * np.power(self.unilateral_kernels[q1].mass, 2) * np.power(omega, 2))

        # Gradient with respect to m, D and C
        for ind_theta in range(3):  # Parameter (m, D or C)
            # Choosing the right gradients for m, omega, gamma1 and gamma2
            if ind_theta == 0:  # Gradient wrt m
                gradThetaM = 1
                gradThetaAlpha = -C / (2 * (m ** 2))
                gradThetaOmega = (C ** 2 - 2 * m * D) / (2 * (m ** 2) * csqrt(4 * m * D - C ** 2))
            if ind_theta == 1:  # Gradient wrt D
                gradThetaM = 0
                gradThetaAlpha = np.zeros((2))
                gradThetaOmega = 1 / csqrt(4 * m * D - C ** 2)
            if ind_theta == 2: # Gradient wrt C
                gradThetaM = 0
                gradThetaAlpha = 1 / (2 * m)
                gradThetaOmega = -C / (2 * m * csqrt(4 * m * D - C ** 2))

        gradThetaGamma1 = gradThetaAlpha + 1j * gradThetaOmega
        gradThetaGamma2 = gradThetaAlpha - 1j * gradThetaOmega
        # Gradient evaluation
        gradThetaGamma = np.array([gradThetaGamma1[0], gradThetaGamma2[0]])
        matGrad = lfmDiagGradientH3(- gamma_m, X, preFactors[0], preExp[:, 1], upsilonDiag[1], gradDiag[1],
                                    diagH[0], gamma_p + gamma_m, gradThetaGamma) \
                  + lfmDiagGradientH3(- gamma_p, X, preFactors[1], preExp[:, 0], upsilonDiag[0], gradDiag[0],
                                      diagH[1], gamma_p + gamma_m, np.hstack([gradThetaGamma[1], gradThetaGamma[0]])) \
                  + lfmDiagGradientH4(X, [gamma_m(gamma_p + gamma_m)], np.hstack([preExp[:, 1], preExp[:, 0]]), upsilonDiag[3],
                                      gradDiag[3], gradThetaGamma) \
                  + lfmDiagGradientH4(X, [gamma_p(gamma_p + gamma_m)], preExp, upsilonDiag[2], gradDiag[2],
                                      np.hstack([gradThetaGamma[1], gradThetaGamma[0]])) \
                  - 2 * (gradThetaM / m + gradThetaOmega / omega) * preKernel
        g[ind_theta] = k0 * sum(sum(matGrad * dL_dKdiag))

        # Gradients with respect to sigma
        if self.unilateral_kernels[q1].isNormalised:
            matGrad = k0 * \
                      (lfmDiagGradientSH3(-gamma_m, sigma2, X, preFactors[0], preExp[:, 1], 1)
                       + lfmDiagGradientSH3(-gamma_p, sigma2, X, preFactors[1], preExp[:, 0], 1)
                       + lfmDiagGradientSH4(gamma_m, sigma2, X, [gamma_m(gamma_p + gamma_m)],
                                            np.hstack(preExp[:, 1], preExp[:, 0]), 1)
                       + lfmDiagGradientSH4(gamma_p, sigma2, X, [gamma_p(gamma_p + gamma_m)], preExp, 1))
        else:
            matGrad = (S ** 2 * csqrt(np.pi) / (8 * m ** 2 * omega ** 2)) \
                      * (preKernel + sigma
                      * (lfmDiagGradientSH3(-gamma_m, sigma2, X, preFactors[0], preExp[:, 1], 1)
                         + lfmDiagGradientSH3(-gamma_p, sigma2, X, preFactors[1], preExp[:, 0], 1)
                         + lfmDiagGradientSH4(gamma_m, sigma2, X, [gamma_m(gamma_p + gamma_m)],
                                              np.hstack([preExp[:, 1], preExp[:, 0]]), 1)
                         + lfmDiagGradientSH4(gamma_p, sigma2, X, [gamma_p(gamma_p + gamma_m)], preExp, 1)))

        g[3]= sum(sum(matGrad * dL_dKdiag)) * (-(sigma ** 3) / 4)

            # Gradients with respect to S

        if self.unilateral_kernels[q1].isNormalised:
            matGrad = (1 / (8 * csqrt(2) * m ** 2 * omega ** 2)) *(preKernel)
        else:
            matGrad = (sigma * csqrt(np.pi) / (8 * m ** 2 * omega ** 2)) *(preKernel)

        g[4] = 2 * S * sum(sum(matGrad * dL_dKdiag))
        g = np.real(g)
        return g

    def update_gradients_diag(self, dL_dKdiag, X):
        self.reset_gradients()
        slices = index_to_slices(X[:, self.index_dim])
        normaliseRegardingToBatchSize = 0
        g = np.zeros((self.output_dim, 5))
        for i in range(len(slices)):
            for k in range(len(slices[i])):
                g[i]=self._update_gradients_diag_wrapper(i, X[slices[i][k], :], dL_dKdiag[slices[i][k]])
                normaliseRegardingToBatchSize += 1
        normalisedg = g/normaliseRegardingToBatchSize
        self.scale.gradient += (normalisedg[3]) * (-2 / np.power(self.scale, 2))
        self.mass.gradient += normalisedg[0]
        self.spring.gradient += normalisedg[1]
        self.damper.gradient += normalisedg[2]
        self.sensitivity.gradient += normalisedg[4]
        return normalisedg






