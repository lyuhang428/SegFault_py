import numpy as np
from scipy.special import factorial2, hyp1f1
from functools import lru_cache
from ase.io import read
from lbeckeGrid import RADII

def sfactorial2(n:int) -> float:
    '''强制要求 (-1)!!=1, 确保对于s轨道lx=ly=lz=0结果正确'''
    return np.where(n <= 0, 1., factorial2(n, exact=True)) # broadcasting
    # return 1. if n <= 0 else factorial2(n)   # scalar only

def boys(n:int, T:float):
    return hyp1f1(n + 0.5, n + 1.5, -T) / (2. * n + 1.)

@lru_cache
def hermite_gauss_e(i:int, j:int, t:int, qx:float, alpha:float, beta:float) -> float:
    '''计算重叠积分，动能积分，矩积分的辅助函数. qx = Ax - Bx'''
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
    
    return (1 / 2. / p)    * hermite_gauss_e(i, j-1, t-1, qx, alpha, beta) + \
           (q * qx / beta) * hermite_gauss_e(i, j-1, t,   qx, alpha, beta) + \
           (t + 1)         * hermite_gauss_e(i, j-1, t+1, qx, alpha, beta)

@lru_cache
def coulomb_auxiliary_r(t:int, u:int, v:int, n:int, p:float, pcx:float, pcy:float, pcz:float, rpc:float) -> float:
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



class Molecule:
    def __init__(self, xyzfile:str, basefile:str='sto-3g'):
        if not xyzfile.endswith('.xyz'):
            raise ValueError('.xyz file format only')
        mol = read(xyzfile, index=0, format='xyz')

        self.xyzfile : str        = xyzfile
        self.basfile : str        = basefile
        self.natom   : int        = len(mol)
        self.symbols : np.ndarray = np.array(mol.get_chemical_symbols())
        self.rms     : np.ndarray = np.array([RADII[symbol] * 1.889726125 / 2 for symbol in self.symbols])
        self.numbers : np.ndarray = mol.get_atomic_numbers()
        self.xyz     : np.ndarray = mol.positions * 1.889726125 # (mol.positions - mol.positions.mean(axis=0)) * 1.88973 # 中心化且转为原子单位
        self.e_nuc   : float      = self.get_nuclear_repulsion_energy()
        self.aos     : list[AO]   = self.get_aos(basefile)
        self.nao     : int        = len(self.aos)
        
        self.rms = np.where(self.symbols == 'H', self.rms * 2, self.rms) # Becke 1988 suggestion
            
    def get_nuclear_repulsion_energy(self) -> float:
        e_nuc = 0.
        for iatom in range(self.natom-1):
            for jatom in range(iatom+1, self.natom):
                e_nuc += self.numbers[iatom] * self.numbers[jatom] / np.linalg.norm(self.xyz[iatom] - self.xyz[jatom])
        return e_nuc

    def get_aos(self, basefile:str) -> "list[AO]":
        '''
        Basis set file format is customized .json file. In `../data/gbs/basis/` directory contains `gbs2json.py` 
        command line tool which converts .gbs to .json. `gbs` are copied from Psi4. 

        Json file can also be read by C++/Fortran with appropriate library, makeing Py-F/C interoperativity eaiser. 
        '''
        import json
        data = json.load(open(f'../data/gbs/basis/{basefile}.json', 'r'))
        aos = []
        for symbol, xyz in zip(self.symbols, self.xyz):
            for shell in data[symbol].keys():
                for subshell in data[symbol][shell]:
                    expns = data[symbol][shell][subshell]['expns']
                    coefs = data[symbol][shell][subshell]['coefs']
                    shells = data[symbol][shell][subshell]['shells']
                    npgf = len(expns)
                    ao = AO(npgf, expns, coefs, shells, xyz)
                    aos.append(ao)
        return aos
    


class AO:
    '''
    一条缩并高斯基函数（原子轨道）

    Parameters
    ----------
    self.npgf : int
        缩并长度

    self.expns : list[float]
        exponents of each pgf
    
    self.coefs : list[float]
        coefficients of each pgf
    
    self.shell : list[int]
        Angular part of AO: x^lx y^ly z^lz. s -> [0,0,0], px -> [1,0,0], py -> [0,1,0], ..., dxx -> [2,0,0], ...
    
    self.center : list[float] | np.ndarray[float]
        xyz coordinate this cgf centers at, in a.u.!

    self.nrfcs : list
        normalization factor for each pgf in a cgf
    '''
    def __init__(self, npgf:int, expns:list, coefs:list, shells:list, center:list):
        '''`center` should be in a.u.'''
        self.npgf   : int        = npgf
        self.expns  : np.ndarray = np.array(expns)
        self.coefs  : np.ndarray = np.array(coefs)
        self.shells : np.ndarray = np.array(shells) # 二维矩阵；对于s,p,dxy,dxz,dyz而言各行相等，对于dx2-y2,d2z2-x2-y2而言至少是两组shell上下堆叠而成
        self.center : np.ndarray = np.array(center) # * 1.88973
        self.nrfcs  : np.ndarray = self.get_nrfcs()

    def get_nrfcs(self) -> np.ndarray:
        '''Normalize pgf.'''
        ltot  = np.sum(self.shells, axis=1) # ; print(self.shells.shape, ltot)
        nrfcs = (2 * self.expns / np.pi)**(3/4)  * (4 * self.expns)**(ltot/2) / \
        np.sqrt(sfactorial2(2 * self.shells[:,0] - 1) * 
                sfactorial2(2 * self.shells[:,1] - 1) * 
                sfactorial2(2 * self.shells[:,2] - 1))
        return nrfcs

    def ao_val_s(self, x:float, y:float, z:float) -> float:
        '''串行计算原子轨道在某点的函数值'''
        shells = np.array([(x-self.center[0])**shell[0] * (y-self.center[1])**shell[1] * (z-self.center[2])**shell[2] for shell in self.shells])
        exp    = np.exp(-self.expns * ((x-self.center[0])**2 + (y-self.center[1])**2 + (z-self.center[2])**2))
        vals   = np.sum(self.coefs * self.nrfcs * shells * exp)
        return vals
    
    def ao_val_p(self, x:np.ndarray, y:np.ndarray, z:np.ndarray) -> np.ndarray:
        '''利用广播机制向量化计算某一原子轨道在多个点的函数值'''
        lx     = np.power.outer(x-self.center[0], self.shells[:,0])
        ly     = np.power.outer(y-self.center[1], self.shells[:,1])
        lz     = np.power.outer(z-self.center[2], self.shells[:,2])
        shells = lx * ly * lz # ; print(shells.T)
        exp    = np.exp(np.outer(-self.expns, (x-self.center[0])**2 + (y-self.center[1])**2 + (z-self.center[2])**2)) # ; print(exp)
        vals   = np.sum((self.coefs * self.nrfcs)[:,None] * shells.T * exp, axis=0) # It works, but why ?
        return vals

    @staticmethod
    def test_ao_val():
        '''测试硬编代码和向量化代码是否给出相同的原子轨道函数值'''
        x = np.random.randn(1000)
        y = np.random.randn(1000)
        z = np.random.randn(1000)

        npgf   = 3, 
        expns  = [0.5033151319E+01, 0.1169596125E+01, 0.3803889600E+00]
        coefs  = [0.1559162750E+00, 0.6076837186E+00, 0.3919573931E+00]
        shells = [[1,1,1],[1,1,1],[1,1,1]]
        # center = [0.564*1.88973, 0.564*1.88973, 0.797616*1.88973]
        center = [0.564*1.889726125, 0.564*1.889726125, 0.797616*1.889726125]
        ao     = AO(npgf, expns, coefs, shells, center)
        nrfcs  = ao.nrfcs
        vals_s = ao.ao_val_s(x, y, z)
        vals_p = ao.ao_val_p(x, y, z)

        fvals = coefs[0] * nrfcs[0] * ((x-center[0]) * (y-center[1]) * (z-center[2])) * np.exp(-expns[0] * ((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2)) + \
                coefs[1] * nrfcs[1] * ((x-center[0]) * (y-center[1]) * (z-center[2])) * np.exp(-expns[1] * ((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2)) + \
                coefs[2] * nrfcs[2] * ((x-center[0]) * (y-center[1]) * (z-center[2])) * np.exp(-expns[2] * ((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2))
        np.testing.assert_allclose(fvals, vals_s, atol=1e-15)
        np.testing.assert_allclose(fvals, vals_p, atol=1e-15)
        print(np.allclose(vals_p, fvals))



class Overlap:
    '''McMurchie-Davidson scheme'''
    # TODO: Add OS scheme for S, T, Dipole (and possibly V)
    def __init__(self, aos:list[AO]):
        self.aos = aos
        self.nao = len(aos)

    @staticmethod
    def _olp_3d(alpha:float, shell1:list, center1:list, beta:float, shell2:list, center2:list) -> float:
        r'''两个未缩并三维高斯函数的重叠积分

        \int (x-xa)^{l1} (y-ya)^{m1} (z-za)^{n1} e^{-\alpha ((x-xa)^2+(y-ya)^2+(z-za)^2)} 
        (x-xb)^{l2} (y-yb)^{m2} (z-zb)^{n2} e^{-\beta ((x-xb)^2+(y-yb)^2+(z-zb)^2)} dxdydz
        '''
        l1, m1, n1 = shell1
        l2, m2, n2 = shell2
        sx = hermite_gauss_e(l1, l2, 0, center1[0] - center2[0], alpha, beta)
        sy = hermite_gauss_e(m1, m2, 0, center1[1] - center2[1], alpha, beta)
        sz = hermite_gauss_e(n1, n2, 0, center1[2] - center2[2], alpha, beta)
        return sx * sy * sz * (np.pi / (alpha + beta))**1.5

    @staticmethod
    def _olp(ao1:AO, ao2:AO) -> float:
        res = 0.
        for nrfc1, coef1, expn1, shell1 in zip(ao1.nrfcs, ao1.coefs, ao1.expns, ao1.shells):
            for nrfc2, coef2, expn2, shell2 in zip(ao2.nrfcs, ao2.coefs, ao2.expns, ao2.shells):
                res += nrfc1 * nrfc2 * coef1 * coef2 * Overlap._olp_3d(expn1, shell1, ao1.center, expn2, shell2, ao2.center)
        return res
    
    def Sij(self) -> np.ndarray:
        sij = np.zeros((self.nao, self.nao))
        for i in range(self.nao-1):
            for j in range(i+1, self.nao):
                sij[i,j] = Overlap._olp(self.aos[i], self.aos[j])
        sij += sij.T
        for i in range(self.nao):
            sij[i,i] = Overlap._olp(self.aos[i], self.aos[i])        

        return sij


class Kinetic:
    def __init__(self, aos:list[AO]):
        self.aos = aos
        self.nao = len(aos)

    @staticmethod
    def _kin_3d(alpha:float, shell1:list, center1:list, beta:float, shell2:list, center2:list) -> float:
        '''kinetic integral over two 3D pgf'''
        l1, m1, n1 = shell1
        l2, m2, n2 = shell2
        term0 = beta * (2 * (l2 + m2 + n2) + 3) *  Overlap._olp_3d(alpha, shell1, center1, beta, shell2, center2)
        term1 = -2. * np.power(beta, 2)         * (Overlap._olp_3d(alpha, shell1, center1, beta, [l2+2, m2, n2], center2) +
                                                   Overlap._olp_3d(alpha, shell1, center1, beta, [l2, m2+2, n2], center2) +
                                                   Overlap._olp_3d(alpha, shell1, center1, beta, [l2, m2, n2+2], center2))
        term2 = -0.5 * (l2 * (l2 - 1) * Overlap._olp_3d(alpha, shell1, center1, beta, [l2-2, m2, n2], center2) +
                        m2 * (m2 - 1) * Overlap._olp_3d(alpha, shell1, center1, beta, [l2, m2-2, n2], center2) +
                        n2 * (n2 - 1) * Overlap._olp_3d(alpha, shell1, center1, beta, [l2, m2, n2-2], center2))
        return term0 + term1 + term2

    @staticmethod
    def _kin(ao1:AO, ao2:AO) -> float:
        res = 0.0
        for nrfc1, coef1, expn1, shell1 in zip(ao1.nrfcs, ao1.coefs, ao1.expns, ao1.shells):
            for nrfc2, coef2, expn2, shell2 in zip(ao2.nrfcs, ao2.coefs, ao2.expns, ao2.shells):
                res += nrfc1 * nrfc2 * coef1 * coef2 * Kinetic._kin_3d(expn1, shell1, ao1.center, expn2, shell2, ao2.center)
        return res
    
    def Tij(self) -> np.ndarray:
        tij = np.zeros((self.nao, self.nao))
        for i in range(self.nao-1):
            for j in range(i+1, self.nao):
                tij[i,j] = Kinetic._kin(self.aos[i], self.aos[j])
        tij += tij.T
        for i in range(self.nao):
            tij[i,i] = Kinetic._kin(self.aos[i], self.aos[i])

        return tij



class External:
    def __init__(self, aos:list[AO], coords:np.ndarray, charges:np.ndarray):
        self.aos     = aos
        self.nao     = len(aos)
        self.coords  = np.array(coords)
        self.charges = np.array(charges)

    @staticmethod
    def _ext_3d(alpha:float, shell1:list, center1:list, beta:float, shell2:list, center2:list, centerC:list) -> float:
        l1, m1, n1 = shell1
        l2, m2, n2 = shell2
        p = alpha + beta
        centerP = (alpha * center1 + beta * center2) / p # ; print(f"{centerP[0]:.15f}    {centerP[1]:.15f}    {centerP[2]:.15f}")
        rpc = np.linalg.norm(centerP - centerC) # ; print(f"{rpc:.15f}")

        res = 0.
        for t in range(l1 + l2 + 1):
            for u in range(m1 + m2 + 1):
                for v in range(n1 + n2 + 1):
                    res += hermite_gauss_e(l1, l2, t, center1[0]-center2[0], alpha, beta) * \
                           hermite_gauss_e(m1, m2, u, center1[1]-center2[1], alpha, beta) * \
                           hermite_gauss_e(n1, n2, v, center1[2]-center2[2], alpha, beta) * \
                           coulomb_auxiliary_r(t, u, v, 0, p, centerP[0]-centerC[0], centerP[1]-centerC[1], centerP[2]-centerC[2], rpc)
        res *= 2. * np.pi / p 
        return res

    @staticmethod
    def _ext(ao1:AO, ao2:AO, centerC:list) -> float:
        res = 0.
        for nrfc1, coef1, expn1, shell1 in zip(ao1.nrfcs, ao1.coefs, ao1.expns, ao1.shells):
            for nrfc2, coef2, expn2, shell2 in zip(ao2.nrfcs, ao2.coefs, ao2.expns, ao2.shells):
                res += nrfc1 * nrfc2 * coef1 * coef2 * External._ext_3d(expn1, shell1, ao1.center, expn2, shell2, ao2.center, centerC)
        return res

    def Vij(self) -> np.ndarray:
        vij = np.zeros((self.nao, self.nao))
        for coord, charge in zip(self.coords, self.charges):
            for i in range(self.nao-1):
                for j in range(i+1, self.nao):
                    vij[i,j] += External._ext(self.aos[i], self.aos[j], coord) * -charge
        vij += vij.T
        for coord, charge in zip(self.coords, self.charges):
            for i in range(self.nao):
                vij[i,i] += External._ext(self.aos[i], self.aos[i], coord) * -charge

        return vij
        


def _sph_d_orb(aos:list[AO]) -> list[AO]:
    '''将笛卡尔d-轨道基函数线性组合成球谐d-轨道基函数(6->5)'''
    if len(aos) != 6:
        raise ValueError('Cartesian d-orbital is six-fold degenerate')
    sph_d_orb = []

    # dx2-y2
    expns  = np.concatenate((aos[0].expns,  aos[3].expns))
    coefs  = np.concatenate((aos[0].coefs*np.sqrt(3.)/2, -aos[3].coefs*np.sqrt(3.)/2))
    shells = np.concatenate((aos[0].shells, aos[3].shells))
    dx2y2  = AO(len(expns), expns, coefs, shells, aos[0].center)

    # d2z2-x2-y2
    expns    = np.concatenate((aos[5].expns,      aos[0].expns,  aos[3].expns))
    coefs    = np.concatenate((aos[5].coefs, -aos[0].coefs/2., -aos[3].coefs/2.))
    shells   = np.concatenate((aos[5].shells,     aos[0].shells, aos[3].shells))
    d2z2x2y2 = AO(len(expns), expns, coefs, shells, aos[0].center)

    sph_d_orb.append(aos[1])   # 2,-2
    sph_d_orb.append(aos[4])   # 2,-1
    sph_d_orb.append(d2z2x2y2) # 2, 0
    sph_d_orb.append(aos[2])   # 2, 1
    sph_d_orb.append(dx2y2)    # 2, 2
    return sph_d_orb

def _sph_f_orb(aos:list[AO]) -> list[AO]:
    r'''将笛卡尔f-轨道基函数线性组合成球谐f-轨道基函数(10->7)'''
    if len(aos) != 10:
        raise ValueError('Cartesian f-orbital is ten-fold degenerate')
    sph_f_orb = []
    
    # 3,-3
    expns = np.concatenate((aos[1].expns, aos[6].expns))
    coefs = np.concatenate((3/4*np.sqrt(2.) * aos[1].coefs, -1/4*np.sqrt(10.) * aos[6].coefs))
    shells = np.concatenate((aos[1].shells, aos[6].shells))
    sph_f_orb.append(AO(len(expns), expns, coefs, shells, aos[0].center, ))

    # 3,-2
    sph_f_orb.append(aos[4]) # fxyz

    # 3,-1
    expns = np.concatenate((aos[8].expns, aos[1].expns, aos[6].expns))
    coefs = np.concatenate((np.sqrt(30.)/5 * aos[8].coefs, -np.sqrt(30.)/20 * aos[1].coefs, -np.sqrt(6.)/4 * aos[6].coefs))
    shells = np.concatenate((aos[8].shells, aos[1].shells, aos[6].shells))
    sph_f_orb.append(AO(len(expns), expns, coefs, shells, aos[0].center, ))

    # 3,0
    expns = np.concatenate((aos[9].expns, aos[2].expns, aos[7].expns))
    coefs = np.concatenate((aos[9].coefs, -3/10*np.sqrt(5.) * aos[2].coefs, -3/10*np.sqrt(5.) * aos[7].coefs))
    shells = np.concatenate((aos[9].shells, aos[2].shells, aos[7].shells))
    sph_f_orb.append(AO(len(expns), expns, coefs, shells, aos[0].center, ))

    # 3,1
    expns = np.concatenate((aos[5].expns, aos[0].expns, aos[3].expns))
    coefs = np.concatenate((np.sqrt(30.)/5 * aos[5].coefs, -np.sqrt(6.)/4 * aos[0].coefs, -np.sqrt(30.)/20 * aos[3].coefs))
    shells = np.concatenate((aos[5].shells, aos[0].shells, aos[3].shells))
    sph_f_orb.append(AO(len(expns), expns, coefs, shells, aos[0].center, ))

    # 3,2
    expns = np.concatenate((aos[2].expns, aos[7].expns))
    coefs = np.concatenate((np.sqrt(3.)/2 * aos[2].coefs, -np.sqrt(3.)/2 * aos[7].coefs))
    shells = np.concatenate((aos[2].shells, aos[7].shells))
    sph_f_orb.append(AO(len(expns), expns, coefs, shells, aos[0].center, ))

    # 3,3
    expns = np.concatenate((aos[0].expns, aos[3].expns))
    coefs = np.concatenate((np.sqrt(10.)/4 * aos[0].coefs, -3/4*np.sqrt(2.) * aos[3].coefs))
    shells = np.concatenate((aos[0].shells, aos[3].shells))
    sph_f_orb.append(AO(len(expns), expns, coefs, shells, aos[0].center, ))

    return sph_f_orb



def benchmark_against_gbasis():
    # GBasis result
    from gbasis.parsers import parse_gbs, make_contractions
    from gbasis.integrals.overlap import overlap_integral
    from gbasis.integrals.kinetic_energy import kinetic_energy_integral
    from gbasis.integrals.nuclear_electron_attraction import nuclear_electron_attraction_integral
    atoms = ['C']
    xyz = np.array([[0., 0., 0.]]) * 1.889726125 # 1.88973
    all_basis_dict = parse_gbs('../data/gbs/sto-3g.gbs')
    basis_sph  = make_contractions(all_basis_dict, atoms, xyz, 'p')
    basis_cart = make_contractions(all_basis_dict, atoms, xyz, 'c')
    gbasis_olp_sph  = overlap_integral(basis_sph)
    gbasis_olp_cart = overlap_integral(basis_cart)
    gbasis_kin_sph = kinetic_energy_integral(basis_sph)
    gbasis_kin_cart = kinetic_energy_integral(basis_cart)
    gbasis_ext_sph = nuclear_electron_attraction_integral(basis_sph, xyz, np.array([6]))
    gbasis_ext_cart = nuclear_electron_attraction_integral(basis_cart, xyz, np.array([6]))
    
    # me
    aos:list[AO] = []
    _d_orbs = []
    _f_orbs = []
    for basis in basis_cart:
        # 以6-31g基组下的碳原子为例，共有1s 2s 2px 2py 2pz 3s 3px 3py 3pz九条基函数，
        # gbasis会将其存储为1s 2s 2p 3s 3p五条基函数，导致对于p轨道的角量子数存储为3x3矩阵形式，
        # 同时s轨道角量子数存储为[[0.0,0]]形式。因此通过循环将角量子数数据抽取出来。
        # 同时，基函数的缩并系数存储为(1xn)形式，使用时将其拉平
        for shell in basis.angmom_components_cart:
            # d-orbital cart->sph 6->5
            if shell.sum() == 2:
                shells = np.array(list(shell) * len(basis.exps)).reshape((-1,3))
                _d_orbs.append(AO(len(basis.exps), basis.exps, basis.coeffs.flatten(), shells, basis.coord))
                if len(_d_orbs) == 6:
                    d_aos = _sph_d_orb(_d_orbs)
                    aos.extend(d_aos)
                    _d_orbs = []
                continue

            # f-orbital cart->sph 10->7
            if shell.sum() == 3:
                shells = np.array(list(shell) * len(basis.exps)).reshape((-1,3))
                _f_orbs.append(AO(len(basis.exps), basis.exps, basis.coeffs.flatten(), shells, basis.coord))
                if len(_f_orbs) == 10:
                    f_aos = _sph_f_orb(_f_orbs)
                    aos.extend(f_aos)
                    _f_orbs = []
                continue

            shells = np.array(list(shell) * len(basis.exps)).reshape((-1,3))
            aos.append(AO(len(basis.exps), basis.exps, basis.coeffs.flatten(), shells, basis.coord))

    olp = Overlap(aos)
    kin = Kinetic(aos)
    ext = External(aos, xyz, [6])
    sij = olp.Sij()
    tij = kin.Tij()
    vij = ext.Vij()

    print(sij.shape, gbasis_olp_sph.shape)
    print(tij.shape, gbasis_kin_sph.shape)
    print(vij.shape, gbasis_ext_sph.shape)
    np.testing.assert_allclose(sij, sij.T, atol=1e-10)
    np.testing.assert_allclose(tij, tij.T, atol=1e-10)
    np.testing.assert_allclose(vij, vij.T, atol=1e-10)

    '''compare and timing'''
    print(abs(sij - gbasis_olp_sph).max())
    print(abs(tij - gbasis_kin_sph).max())
    print(abs(vij - gbasis_ext_sph).max())


def benchmark_against_gbasis2():
    # gbasis
    from gbasis.parsers import parse_gbs, make_contractions
    from gbasis.integrals.overlap import overlap_integral
    from gbasis.integrals.kinetic_energy import kinetic_energy_integral
    from gbasis.integrals.nuclear_electron_attraction import nuclear_electron_attraction_integral
    atoms = ['O']
    xyz = np.array([[0., 0., 0.]]) * 1.88973
    all_basis_dict = parse_gbs('../data/gbs/sto-3g.gbs')
    basis_sph  = make_contractions(all_basis_dict, atoms, xyz, 'p')
    basis_cart = make_contractions(all_basis_dict, atoms, xyz, 'c')
    gbasis_olp_sph  = overlap_integral(basis_sph)
    gbasis_olp_cart = overlap_integral(basis_cart)
    gbasis_kin_sph = kinetic_energy_integral(basis_sph)
    gbasis_kin_cart = kinetic_energy_integral(basis_cart)
    gbasis_ext_sph = nuclear_electron_attraction_integral(basis_sph, xyz, np.array([8]))
    gbasis_ext_cart = nuclear_electron_attraction_integral(basis_cart, xyz, np.array([8]))

    # me
    mol = Molecule('../data/xyz/o.xyz')
    olp = Overlap(mol.aos)
    kin = Kinetic(mol.aos)
    ext = External(mol.aos, mol.xyz, mol.numbers)
    Sij = olp.Sij()
    Tij = kin.Tij()
    Vij = ext.Vij()

    print(abs(Sij - gbasis_olp_cart).max(), abs(Sij - gbasis_olp_sph).max(), )
    print(abs(Tij - gbasis_kin_cart).max(), abs(Tij - gbasis_kin_sph).max(), )
    print(abs(Vij - gbasis_ext_cart).max(), abs(Vij - gbasis_ext_sph).max(), )
    

def __test_boys():
    n = np.random.randint(low=-10, high=50, size=1000)
    T = np.random.random(1000) * 20
    res = boys(n, T)
    with open('/home/lyh/dft/ldft2025-11-21/test/boys-testdata.dat', 'w') as fp:
        for nn, tt, rr in zip(n, T, res):
            fp.writelines(f"{nn:<10d}    {tt:>20.15f}    {rr:>20.15f}")
            fp.writelines('\n')

def __test_oeintegral():
    np.set_printoptions(10)
    mol = Molecule('../data/xyz/co2.xyz', 'sto-3g')
    olp = Overlap(mol.aos)
    kin = Kinetic(mol.aos)
    ext = External(mol.aos, mol.xyz, mol.numbers)

    sij_cpp = np.loadtxt('../../../.sij.dat', dtype=float)
    tij_cpp = np.loadtxt('../../../.tij.dat', dtype=float)
    vij_cpp = np.loadtxt('../../../.vij.dat', dtype=float)
    np.testing.assert_allclose(sij:=olp.Sij(), sij_cpp, atol=1e-15); print("sij pass")
    np.testing.assert_allclose(tij:=kin.Tij(), tij_cpp, atol=1e-15); print("tij pass")
    np.testing.assert_allclose(vij:=ext.Vij(), vij_cpp, atol=1e-15); print("vij pass")



if __name__ == '__main__':
    mol = Molecule("../data/xyz/co2.xyz", "cc-pvdz")
    # for ao in mol.aos:
    #     print("exponents:")
    #     for expn in ao.expns:
    #         print(expn, end="  ")
    #     print()

    #     print("coefficients: ")
    #     for (coef, nrfc) in zip(ao.coefs, ao.nrfcs):
    #         print(coef * nrfc, end="  ")
    #     print()
    x = np.arange(-10., 10., 0.01)
    y = np.arange(-10., 10., 0.01); y *= 1.5
    z = np.arange(-10., 10., 0.01); z -= 2.459
    
    data = []
    aos = mol.aos.copy()
    for ao in aos:
        data.append(ao.ao_val_p(x, y, z))
    print(np.array(data).shape)
    np.save('./tmp.npy', np.array(data))
