# This file is part of homemade-hf, a simple Hartree-Fock implementation.
# This file contains utility functions and constants used throughout the project. 
# Author: lyh


import numpy as np
from scipy.special import factorial, factorial2, gamma, gammaincc, hyp1f1
from typing import Any


class IOStream:
    '''C++ style output: cout << "somt thnig" << endl
    
    good for people who's tierd of print() and want to use cout style
    '''
    def __init__(self):
        pass

    def __lshift__(self, other:Any) -> "IOStream":
        print(other, end='')
        return self

COUT = IOStream()
ENDL = '\n'


class MathConst:
    euler   = 0.57721566490153286060651209 # Euler-Mascheroni constant
    pi      = 3.14159265358979323846264338
    twopi   = 6.28318530717958647692528677
    fourpi  = 12.56637061435917295385057353
    halfpi  = 1.57079632679489661923132169
    rootpi  = 1.77245385090551602729816748
    pi2     = 9.8696044010893586
    irootpi = 0.56418958354775628694807945
    norm_pi = 0.4237772081237576 # (1/pi)^(3/4), primitive gaussian normalization factor

    
class PhysicConst:
    c_light = 299792458. # Speed of light in vacuum [m/s]
    c_light_au = 137.035999679 # Speed of light in vacuum, in atomic units (=1/a_fine)
    mu_perm = 4. * MathConst.pi * 1.0E-7 # Magnetic constant or permeability of vacuum [N/A**2]
    permittivity = 1. / (mu_perm * c_light**2) # Electric constant or permittivity of vacuum [F/m]
    h_planck = 6.62606896E-34 # Planck constant [J*s]
    hbar = h_planck / MathConst.twopi # reduced Plancl const
    e_charge = 1.602176487E-19 # Elementary charge [C]
    e_mass = 9.10938215E-31 # Electron mass [kg]
    p_mass = 1.672621637E-27 # Proton mass [kg]
    e_gfactor = -2.0023193043622 # Electron g factor [ ]
    a_fine = 7.2973525376E-3 # Fine-structure constant
    rydberg = 10973731.568527 # Rydberg constant [1/m]
    n_avogadro = 6.02214179E+23 # Avogadro constant [1/mol]
    boltzmann = 1.3806504E-23 # Boltzmann constant [J/K]
    a_bohr = 0.52917720859E-10 # Bohr [m]


class _AtomInfo:
    def __init__(self):
        self.name            : str | None = None
        self.symbol          : str | None = None
        self.number          : int | None = None           # nuclear charge
        self.amass           : float | None = None         # averaged mass
        self.mass            : float | None = None         # mass of most abundant isotope
        self.covalent_radius : float | None = None         # angstrom
        self.vdw_radius      : float | None = None         # angstrom
        self.e_config        : list[int] = [0,0,0,0]       # ne in s,p,d,f orbital
        self.eht_param       : list[float] = [0.,0.,0.,0.] # ionization energy in s,p,d,f orbital, eV


class PeriodicTable:
    def __init__(self):
        self.ptable = [_AtomInfo()] * 106 # 105 elements + ghost
        # TODO: utils.f90 ptable module contents move here, copied from cp2k

    
def timeit(f:callable) -> callable:
    def wrapper(*args, **kwargs) -> Any:
        import time
        begin = time.time()
        res = f(*args, **kwargs)
        print(f"{f.__name__} took {time.time() - begin:.6f} sec")
        return res
    return wrapper



def safe_factorial2(n:int) -> float:
    '''scipy.special.factorial2(-1) is undefined, but we need it to be 1 for s-orbital [0,0,0]'''
    return 1. if n <= 0 else factorial2(n)



def _boys(m:int, x:float) -> float:
    '''boys function, but very inaccurate in (0,50)'''
    if x < 0:
        raise ValueError('x must be non-negative')
    
    if x <= 1e-15:
        return 1. / (2. * m + 1.) # if x = 0 return 1/(2m+1)
    elif x >= 50.:
        return factorial2(2 * m - 1) / 2.**(m + 1) * MathConst.rootpi / np.sqrt(x**(2 * m + 1))
    else:
        # this range is not accurate
        return 0.5 * x**(-0.5 - m) * (gamma(0.5 + m) - gammaincc(0.5 + m, x))
    

def boys(n:int, T:float):
    '''analytic evaluation of Boys function, no asymptotic approximation'''
    return hyp1f1(n + 0.5, n + 1.5, -T) / (2. * n + 1.)
    


if __name__ == '__main__':
    pass


