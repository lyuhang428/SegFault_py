# This file is part of homemade-hf, a simple Hartree-Fock implementation.
# This file is a pure Python implementation of the McMurchie-Davidson scheme for evaluating molecular integrals over Gaussian functions.
# This implementation is the slowest among mmd, mmd2, mmd3. But it's straightforward, focusing on clarity rather than performance.
# Author: lyh


import numpy as np
from molecule import Molecule, ContractedGaussianFunction
from utils import boys
from functools import lru_cache


@lru_cache
def hermite_gauss_e(i, j, t, qx, alpha, beta) -> float:
    '''qx = Ax - Bx'''
    p = alpha + beta
    q = alpha * beta / p

    if (t < 0) or (t > (i + j)):
        return 0.
    elif i == j == t == 0:
        return np.exp(-q * qx * qx)
    elif j == 0:
        return (1 / 2. / p)     * hermite_gauss_e(i-1, j, t-1, qx, alpha, beta) - \
               (q * qx / alpha) * hermite_gauss_e(i-1, j, t,   qx, alpha, beta) + \
               (t + 1)          * hermite_gauss_e(i-1, j, t+1, qx, alpha, beta)
    else:
        return (1 / 2. / p)    * hermite_gauss_e(i, j-1, t-1, qx, alpha, beta) + \
               (q * qx / beta) * hermite_gauss_e(i, j-1, t,   qx, alpha, beta) + \
               (t + 1)         * hermite_gauss_e(i, j-1, t+1, qx, alpha, beta)

def overlap_3d(alpha, shell1:list, center1:list, beta, shell2:list, center2:list) -> float:
    '''overlap over two 3D pgf'''
    l1, m1, n1 = shell1
    l2, m2, n2 = shell2
    sx = hermite_gauss_e(l1, l2, 0, center1[0] - center2[0], alpha, beta)
    sy = hermite_gauss_e(m1, m2, 0, center1[1] - center2[1], alpha, beta)
    sz = hermite_gauss_e(n1, n2, 0, center1[2] - center2[2], alpha, beta)
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

@lru_cache
def coulomb_auxiliary_r(t, u, v, n, p, pcx, pcy, pcz, rpc) -> float:
    '''
    Parameters
    ----------

    t, u, v : int
        Coulomb Hermite derivative in x, y, z direction
    
    n : int
        order of Boys function
    
    p : float
        p = alpha + beta

    pcx, pcy, pcz : float
        px - cx, py - cy, pz - cz, where C is nuclear coordinate, P is gaussian product center
        px = (alpha * ax + beta * bx) / (alpha + beta)

    rpc : float
        |P - C|
    '''
    _T = p * rpc * rpc
    res = 0.
    if t == u == v == 0:
        res += np.power(-2. * p, n) * boys(n, _T)
    elif t == u == 0:
        if v >= 2:
            res += (v - 1) * coulomb_auxiliary_r(t, u, v-2, n+1, p, pcx, pcy, pcz, rpc)
        res += pcz * coulomb_auxiliary_r(t, u, v-1, n+1, p, pcx, pcy, pcz, rpc)
    elif t == 0:
        if u >= 2:
            res += (u - 1) * coulomb_auxiliary_r(t, u-2, v, n+1, p, pcx, pcy, pcz, rpc)
        res += pcy * coulomb_auxiliary_r(t, u-1, v, n+1, p, pcx, pcy, pcz, rpc)
    else:
        if t >= 2:
            res += (t - 1) * coulomb_auxiliary_r(t-2, u, v, n+1, p, pcx, pcy, pcz, rpc)
        res += pcx * coulomb_auxiliary_r(t-1, u, v, n+1, p, pcx, pcy, pcz, rpc)
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
                res += hermite_gauss_e(l1, l2, t, center1[0]-center2[0], alpha, beta) * \
                       hermite_gauss_e(m1, m2, u, center1[1]-center2[1], alpha, beta) * \
                       hermite_gauss_e(n1, n2, v, center1[2]-center2[2], alpha, beta) * \
                       coulomb_auxiliary_r(t, u, v, 0, p, centerP[0]-centerC[0], centerP[1]-centerC[1], centerP[2]-centerC[2], rpc)
    res *= 2. * np.pi / p 
    return res

def potential(cgf1:ContractedGaussianFunction, cgf2:ContractedGaussianFunction, centerC:list) -> float:
    res = 0.
    for i1, coef1 in enumerate(cgf1.coefs):
        for i2, coef2 in enumerate(cgf2.coefs):
            res += cgf1.normalize_factors[i1] * cgf2.normalize_factors[i2] * coef1 * coef2 * \
                     potential_3d(cgf1.expns[i1], cgf1.shell, cgf1.center, cgf2.expns[i2], cgf2.shell, cgf2.center, centerC)
    return res

def eri_3d(alpha, shell1:list, center1:list, 
           beta,  shell2:list, center2:list, 
           gamma, shell3:list, center3:list, 
           delta, shell4:list, center4:list) -> float:
    l1, m1, n1 = shell1
    l2, m2, n2 = shell2
    l3, m3, n3 = shell3
    l4, m4, n4 = shell4
    p = alpha + beta
    q = gamma + delta
    PQ = p * q / (p + q)
    centerP:list = (alpha * center1 + beta  * center2) / p
    centerQ:list = (gamma * center3 + delta * center4) / q
    rpq = np.linalg.norm(centerP - centerQ)

    res = 0.
    for t in range(l1 + l2 + 1):
        for u in range(m1 + m2 + 1):
            for v in range(n1 + n2 + 1):
                for tau in range(l3 + l4 + 1):
                    for nu in range(m3 + m4 + 1):
                        for phi in range(n3 + n4 + 1):
                            res += hermite_gauss_e(l1, l2, t,   center1[0]-center2[0], alpha, beta)  * \
                                   hermite_gauss_e(m1, m2, u,   center1[1]-center2[1], alpha, beta)  * \
                                   hermite_gauss_e(n1, n2, v,   center1[2]-center2[2], alpha, beta)  * \
                                   hermite_gauss_e(l3, l4, tau, center3[0]-center4[0], gamma, delta) * \
                                   hermite_gauss_e(m3, m4, nu , center3[1]-center4[1], gamma, delta) * \
                                   hermite_gauss_e(n3, n4, phi, center3[2]-center4[2], gamma, delta) * \
                                   np.power(-1, tau + nu + phi) * \
                                   coulomb_auxiliary_r(t+tau, u+nu, v+phi, 0, PQ, centerP[0]-centerQ[0], centerP[1]-centerQ[1], centerP[2]-centerQ[2], rpq)
    res *= 2. * np.power(np.pi, 2.5) / (p * q * np.sqrt(p + q))
    return res

def eri(cgf1:ContractedGaussianFunction, cgf2:ContractedGaussianFunction, cgf3:ContractedGaussianFunction, cgf4:ContractedGaussianFunction) -> float:
    res = 0.
    for i1, coef1 in enumerate(cgf1.coefs):
        for i2, coef2 in enumerate(cgf2.coefs):
            for i3, coef3 in enumerate(cgf3.coefs):
                for i4, coef4 in enumerate(cgf4.coefs):
                    res += cgf1.normalize_factors[i1] * cgf2.normalize_factors[i2] * cgf3.normalize_factors[i3] * cgf4.normalize_factors[i4] * \
                           coef1 * coef2 * coef3 * coef4 * \
                           eri_3d(cgf1.expns[i1], cgf1.shell, cgf1.center,
                                  cgf2.expns[i2], cgf2.shell, cgf2.center,
                                  cgf3.expns[i3], cgf3.shell, cgf3.center,
                                  cgf4.expns[i4], cgf4.shell, cgf4.center)
    return res



if __name__ == '__main__':
    pass

