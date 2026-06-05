# This file is part of homemade-hf, a simple Hartree-Fock implementation.
# This file uses `ctypes` to wrap C code for the McMurchie-Davidson scheme for evaluating molecular integrals over Gaussian functions.
# This implementation is faster than mmd.py, but slower than mmd2.py.
# The C code is straightforward and easy to read, while moderately performance improved. 
# The C code is not OpenMP parallelized. 
# Author: lyh

import numpy as np
import ctypes
from molecule import Molecule, ContractedGaussianFunction

lib = ctypes.CDLL('./core/libwrapper.so')

lib.boys.argtypes = [ctypes.c_int, ctypes.c_double]
lib.boys.restype = ctypes.c_double

lib.hermite_gauss_e.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double]
lib.hermite_gauss_e.restype = ctypes.c_double

lib.coulomb_auxiliary_r.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                                    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
lib.coulomb_auxiliary_r.restype = ctypes.c_double

lib.eri_3d.argtypes = [ctypes.c_double, ctypes.c_int * 3, ctypes.c_double * 3, 
                       ctypes.c_double, ctypes.c_int * 3, ctypes.c_double * 3, 
                       ctypes.c_double, ctypes.c_int * 3, ctypes.c_double * 3, 
                       ctypes.c_double, ctypes.c_int * 3, ctypes.c_double * 3]
lib.eri_3d.restype = ctypes.c_double



def overlap_3d(alpha, shell1:list, center1:list, beta, shell2:list, center2:list) -> float:
    '''overlap over two 3D pgf'''
    l1, m1, n1 = shell1
    l2, m2, n2 = shell2
    sx = lib.hermite_gauss_e(l1, l2, 0, center1[0] - center2[0], alpha, beta)
    sy = lib.hermite_gauss_e(m1, m2, 0, center1[1] - center2[1], alpha, beta)
    sz = lib.hermite_gauss_e(n1, n2, 0, center1[2] - center2[2], alpha, beta)
    return sx * sy * sz * np.power(np.pi / (alpha + beta), 1.5) 

def overlap(cgf1:ContractedGaussianFunction, cgf2:ContractedGaussianFunction) -> float:
    res = 0.0
    for i1, coef1 in enumerate(cgf1.coefs):
        for i2, coef2 in enumerate(cgf2.coefs):
            res += cgf1.normalize_factors[i1] * cgf2.normalize_factors[i2] * coef1 * coef2 * \
                   overlap_3d(cgf1.expns[i1], cgf1.shell, cgf1.center, cgf2.expns[i2], cgf2.shell, cgf2.center)
    return res

def kinetic_3d(alpha, shell1:list, center1:list, beta, shell2:list, center2:list) -> float:
    '''kinetic integral over two 3D pgf'''
    l1, m1, n1 = shell1
    l2, m2, n2 = shell2
    term0 = beta * (2 * (l2 + m2 + n2) + 3) * overlap_3d(alpha, shell1, center1, beta, shell2, center2)
    term1 = -2. * np.power(beta, 2) * (overlap_3d(alpha, shell1, center1, beta, [l2+2, m2, n2], center2) +
                                       overlap_3d(alpha, shell1, center1, beta, [l2, m2+2, n2], center2) +
                                       overlap_3d(alpha, shell1, center1, beta, [l2, m2, n2+2], center2))
    term2 = -0.5 * (l2 * (l2 - 1) * overlap_3d(alpha, shell1, center1, beta, [l2-2, m2, n2], center2) +
                    m2 * (m2 - 1) * overlap_3d(alpha, shell1, center1, beta, [l2, m2-2, n2], center2) +
                    n2 * (n2 - 1) * overlap_3d(alpha, shell1, center1, beta, [l2, m2, n2-2], center2))
    return term0 + term1 + term2

def kinetic(cgf1:ContractedGaussianFunction, cgf2:ContractedGaussianFunction) -> float:
    res = 0.0
    for i1, coef1 in enumerate(cgf1.coefs):
        for i2, coef2 in enumerate(cgf2.coefs):
            res += cgf1.normalize_factors[i1] * cgf2.normalize_factors[i2] * coef1 * coef2 * \
                   kinetic_3d(cgf1.expns[i1], cgf1.shell, cgf1.center, cgf2.expns[i2], cgf2.shell, cgf2.center)
    return res


def potential_3d(alpha, shell1:list, center1:list, beta, shell2:list, center2:list, centerC:list) -> float:
    '''
    Parameters
    ----------

    centerC : list
        coordinate of nuclear
    '''
    l1, m1, n1 = shell1
    l2, m2, n2 = shell2
    p = alpha + beta
    centerP = (alpha * center1 + beta * center2) / p
    rpc = np.linalg.norm(centerP - centerC)

    res = 0.0
    for t in range(l1 + l2 + 1):
        for u in range(m1 + m2 + 1):
            for v in range(n1 + n2 + 1):
                res += lib.hermite_gauss_e(l1, l2, t, center1[0]-center2[0], alpha, beta) * \
                       lib.hermite_gauss_e(m1, m2, u, center1[1]-center2[1], alpha, beta) * \
                       lib.hermite_gauss_e(n1, n2, v, center1[2]-center2[2], alpha, beta) * \
                       lib.coulomb_auxiliary_r(t, u, v, 0, p, centerP[0]-centerC[0], centerP[1]-centerC[1], centerP[2]-centerC[2], rpc)
    res *= 2. * np.pi / p 
    return res

def potential(cgf1:ContractedGaussianFunction, cgf2:ContractedGaussianFunction, centerC:list) -> float:
    res = 0.
    for i1, coef1 in enumerate(cgf1.coefs):
        for i2, coef2 in enumerate(cgf2.coefs):
            res += cgf1.normalize_factors[i1] * cgf2.normalize_factors[i2] * coef1 * coef2 * \
                     potential_3d(cgf1.expns[i1], cgf1.shell, cgf1.center, cgf2.expns[i2], cgf2.shell, cgf2.center, centerC)
    return res


def eri(cgf1:ContractedGaussianFunction, cgf2:ContractedGaussianFunction, cgf3:ContractedGaussianFunction, cgf4:ContractedGaussianFunction) -> float:
    res = 0.
    for i1, coef1 in enumerate(cgf1.coefs):
        for i2, coef2 in enumerate(cgf2.coefs):
            for i3, coef3 in enumerate(cgf3.coefs):
                for i4, coef4 in enumerate(cgf4.coefs):
                    res += cgf1.normalize_factors[i1] * cgf2.normalize_factors[i2] * cgf3.normalize_factors[i3] * cgf4.normalize_factors[i4] * \
                           coef1 * coef2 * coef3 * coef4 * \
                           lib.eri_3d(cgf1.expns[i1], (ctypes.c_int * 3)(*cgf1.shell), (ctypes.c_double * 3)(*cgf1.center),
                                      cgf2.expns[i2], (ctypes.c_int * 3)(*cgf2.shell), (ctypes.c_double * 3)(*cgf2.center),
                                      cgf3.expns[i3], (ctypes.c_int * 3)(*cgf3.shell), (ctypes.c_double * 3)(*cgf3.center),
                                      cgf4.expns[i4], (ctypes.c_int * 3)(*cgf4.shell), (ctypes.c_double * 3)(*cgf4.center))
    return res



if __name__ == '__main__':
    pass



