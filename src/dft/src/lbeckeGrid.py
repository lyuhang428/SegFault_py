import numpy as np
from llebedev import lebedev_rule # or from scipy.integrate import lebedev_rule

# References
# [1] Slater, John C. "Atomic radii in crystals." The Journal of Chemical Physics 41.10 (1964): 3199-3204.
# [2] Becke 1988
# Slater paper suggests 0.25 for H, Becke 1988 paper suggests 0.35
# unit in angstrom
RADII = { 'H': 0.35, 'He': 2.00, 'Li': 1.45, 'Be': 1.05,  'B': 0.85,  
          'C': 0.70,  'N': 0.65,  'O': 0.60,  'F': 0.50, 'Ne': 2.25, 
         'Na': 1.80, 'Mg': 1.50, 'Al': 1.25, 'Si': 1.10,  'P': 1.00, 
          'S': 1.00, 'Cl': 1.00, 'Ar': 2.50,  'K': 2.20, 'Ca': 1.80, 
         'Sc': 1.60, 'Ti': 1.40,  'V': 1.35, 'Cr': 1.40, 'Mn': 1.40, 
         'Fe': 1.40, 'Co': 1.35, 'Ni': 1.35, 'Cu': 1.35, 'Zn': 1.35, 
         'Ga': 1.30, 'Ge': 1.25, 'As': 1.15, 'Se': 1.15, 'Br': 1.15, 
         'Kr': 2.75, 'Rb': 2.35, 'Sr': 2.00,  'Y': 1.80, 'Zr': 1.55, 
         'Nb': 1.45, 'Mo': 1.45, 'Tc': 1.35, 'Ru': 1.30, 'Rh': 1.35, 
         'Pd': 1.40, 'Ag': 1.60, 'Cd': 1.55, 'In': 1.55, 'Sn': 1.45, 
         'Sb': 1.45, 'Te': 1.40,  'I': 1.40, 'Xe': 3.00, 'Cs': 2.60, 
         'Ba': 2.15, 'La': 1.95, 'Ce': 1.85, 'Pr': 1.85, 'Nd': 1.85, 
         'Pm': 1.85, 'Sm': 1.85, 'Eu': 1.85, 'Gd': 1.80, 'Tb': 1.75, 
         'Dy': 1.75, 'Ho': 1.75, 'Er': 1.75, 'Tm': 1.75, 'Yb': 1.75, 
         'Lu': 1.75, 'Hf': 1.55, 'Ta': 1.45,  'W': 1.35, 'Re': 1.35, 
         'Os': 1.30, 'Ir': 1.35, 'Pt': 1.35, 'Au': 1.35, 'Hg': 1.50, 
         'Tl': 1.90, 'Pb': 1.80, 'Bi': 1.60, 'Po': 1.90, 'At': 1.65, 
         'Rn': 3.25, 'Fr': 2.80, 'Ra': 2.15, 'Ac': 1.95, 'Th': 1.80, 
         'Pa': 1.80,  'U': 1.75, 'Np': 1.75, 'Pu': 1.75, 'Am': 1.75,  }

def timeit(f):
    def wrapper(*args, **kwargs):
        from time import time
        begin = time()
        res = f(*args, **kwargs)
        end = time()
        print(f'\n{f.__name__} took {end-begin:>.6f} sec')
        return res
    return wrapper


# Becke1988 switching function
# -1 <= u <= 1
p  = lambda u: 3 / 2 * u - 1 / 2 * u**3
f1 = lambda u: p(u)                ; s1 = lambda u: 0.5 * (1 - f1(u))
f2 = lambda u: p(p(u))             ; s2 = lambda u: 0.5 * (1 - f2(u))
f3 = lambda u: p(p(p(u)))          ; s3 = lambda u: 0.5 * (1 - f3(u))
f4 = lambda u: p(p(p(p(u))))       ; s4 = lambda u: 0.5 * (1 - f4(u))
f5 = lambda u: p(p(p(p(p(u)))))    ; s5 = lambda u: 0.5 * (1 - f5(u))
f6 = lambda u: p(p(p(p(p(p(u)))))) ; s6 = lambda u: 0.5 * (1 - f6(u))

def __visualize_switching_functions():
    import matplotlib.pyplot as plt
    u = np.linspace(-1, 1, 201) # -1 <= u <= 1
    plt.plot(u, f1(u), '--', label='f1')
    plt.plot(u, f2(u), '--', label='f2')
    plt.plot(u, f3(u), '--', label='f3')
    plt.plot(u, f4(u), '--', label='f4')
    plt.plot(u, f5(u), '--', label='f5')
    plt.plot(u, f6(u), '--', label='f6')

    plt.plot(u, s1(u), label='s1')
    plt.plot(u, s2(u), label='s2')
    plt.plot(u, s3(u), label='s3')
    plt.plot(u, s4(u), label='s4')
    plt.plot(u, s5(u), label='s5')
    plt.plot(u, s6(u), label='s6')
    plt.xlim((-2, 2))
    plt.ylim((-1.5, 1.5))
    plt.legend()
    plt.tight_layout()
    plt.show()


def gaussCheby2(norder:int=32, rm:float=1.) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r'''高斯-切比雪夫求积（第二类）. 权重函数和雅可比行列式已包含在返回的权重值中
    
    Example
    -------
    >>> f = lambda x: x**2 * np.exp(-0.5 * x**2) # \int_0^\infty f^2 dx = 3 \sqrt{\pi} / 8 
    >>> val = 0.6646701940895685
    >>> xcheb, rcheb, wcheb = gaussCheby2()
    >>> integral = np.sum(f(rcheb)**2 * wcheb)
    >>> print(f'Absolute error: {integral-val:>20.16f}') # Absolute error:  -0.0000002505416914
    '''
    z = np.arange(1, norder+1)           # 网格z均匀间隔, 對 ChebyshevII 網格進行有限差分時分母 h=1
    x = np.cos(z / (norder + 1) * np.pi) # z ∈ [1, norder] ↦ x ∈ (1, -1)
    r = rm * (1 + x) / (1 - x)           # x ∈ (1, -1) ↦ r ∈ (∞, 0)
    w = np.pi / (norder + 1) * np.sin(z / (norder + 1) * np.pi)**2 / np.sqrt(1 - x**2) * 2 * rm / (1 - x)**2
    return x, r, w

def treutler():
    '''
    Reference
    ---------
    Oliver Treutler and Reinhart Ahlrichs, J. Chem. Phys. 102 (1), 1 January 1995
    '''
    ...


class BeckeFuzzyCell:
    '''
    Reference
    ---------
    Becke 1988. J. Chem. Phys. 88, 2547-2553 (1988)
    '''
    def __init__(self, symbols:np.ndarray, rms:np.ndarray, numbers:np.ndarray, xyz:np.ndarray, *, 
                 ncheb:int=32, nleb:int=17, k:int=3, biased:bool=True):
        '''
        先實例化`Molecule`對象, 使用該對象的屬性實例化`BeckeFuzzyCell`對象

        Notes
        -----
        `self.chebs`, `self.xleb`, `self.wleb` are not `None` only after `self.build_grid` is called

        Arguments
        ---------
        symbols : np.ndarray[str]
            Element characters
        rms : np.ndarray[float]
            Bragg-Slater radii / 2 * 1.88973. For H not divided by 2
        numbers : np.ndarray[int]
            Nuclear charges
        xyz : np.ndarray
            Centeralized molecular coordinats in Bohr
        ncheb : int = 32
            Number of Gauss-ChebyshevII radial grid shell
        nleb : int = 17
            Order of Lebedev angular grid
        k : int = 3
            Order of switching function
        biased : bool = True
            If assign different radii when build up molecular grids
        '''
        self.symbols : np.ndarray = symbols
        self.rms     : np.ndarray = rms
        self.numbers : np.ndarray = numbers
        self.xyz     : np.ndarray = xyz
        self.natom   : int        = len(symbols)
        self.ncheb   : int        = ncheb
        self.nleb    : int        = nleb
        self.k       : int        = k
        self.biased  : bool       = biased
        # self.chebs   : np.ndarray = None # (natom, ncheb, (ri,wi))
        self.zcheb   : np.ndarray = None # (natom,ncheb)
        self.xcheb   : np.ndarray = None # (natom,ncheb)
        self.rcheb   : np.ndarray = None # (natom,ncheb)
        self.wcheb   : np.ndarray = None # (natom,ncheb)
        self.xleb    : np.ndarray = None # (3, nang)
        self.wleb    : np.ndarray = None # (nang,)
        self.weight  : np.ndarray = None # (natom,nradxnang)

    def get_weight_p(self, grid:np.ndarray, ) -> np.ndarray:
        '''计算所有原子在所有网格点上的权重
        
        Arguments
        ---------
        grid : np.ndarray (N,3)
            Grid point coordinate in real space.

        Return
        ------
        p : np.ndarray (M_atoms, N_grid)
            Summation along axis=0 should give exactly unity. 
        '''

        dist = np.linalg.norm(self.xyz - grid[:,None,:], axis=2) # (ngrid, matom)
        p = np.zeros((self.natom, len(grid)))

        for iatom in range(self.natom):
            s = 1.
            for jatom in range(self.natom):
                if jatom == iatom:
                    continue
                ri = dist[:,iatom]
                rj = dist[:,jatom]
                rij = np.linalg.norm(self.xyz[iatom] - self.xyz[jatom])
                muij = (ri - rj) / rij

                if self.biased:
                    chi = RADII[self.symbols[iatom]] / RADII[self.symbols[jatom]]
                    uij = (chi - 1) / (chi + 1)
                    aij = uij / (uij**2 - 1)
                    if aij > 0.5:
                        aij = 0.5
                    elif aij < -0.5:
                        aij = -0.5
                    nuij = muij + aij * (1 - muij**2)
                else:
                    nuij = muij

                if self.k == 1:
                    sij = s1(nuij)
                elif self.k == 2:
                    sij = s2(nuij)
                elif self.k == 3:
                    sij = s3(nuij)
                elif self.k == 4:
                    sij = s4(nuij)
                elif self.k == 5:
                    sij = s5(nuij)
                elif self.k == 6:
                    sij = s6(nuij)
                else:
                    raise ValueError('k > 6 not possible')

                s *= sij
            p[iatom] = s
        p /= p.sum(axis=0)
        return p
    
    def get_weight_s(self, x:float, y:float, z:float, ) -> list[float]:
        '''計算一個網格點上各原子的權重
        
        Arguments
        ---------
        x, y, z : float
            Grid coordinate in 3D real space

        Return
        ------
        p : list[float]
            Weight of all atmos at given grid point (x,y,z)
        '''
        r = np.array([x,y,z])
        p = np.zeros((self.natom,))
        for iatom in range(self.natom):
            s = 1.
            for jatom in range(self.natom):
                if jatom == iatom:
                    continue
                ri = np.linalg.norm(r - self.xyz[iatom])
                rj = np.linalg.norm(r - self.xyz[jatom])
                rij = np.linalg.norm(self.xyz[iatom] - self.xyz[jatom])
                muij = (ri - rj) / rij

                if self.biased:
                    chi = RADII[self.symbols[iatom]] / RADII[self.symbols[jatom]]
                    uij = (chi - 1) / (chi + 1)
                    aij = uij / (uij**2 - 1)
                    if aij > 0.5:
                        aij = 0.5
                    elif aij < -0.5:
                        aij = -0.5
                    nuij = muij + aij * (1 - muij**2)
                else:
                    nuij = muij

                if self.k == 1:
                    sij = s1(nuij)
                elif self.k == 2:
                    sij = s2(nuij)
                elif self.k == 3:
                    sij = s3(nuij)
                elif self.k == 4:
                    sij = s4(nuij)
                elif self.k == 5:
                    sij = s5(nuij)
                elif self.k == 6:
                    sij = s6(nuij)
                else:
                    raise ValueError('k > 6 not possible')
                s *= sij
            p[iatom] = s
        p /= p.sum()
        return p

    def build_grid(self) -> tuple[np.ndarray, np.ndarray]:
        '''構建以原子爲中心的Becke網格

        Return
        ------
        grid : np.ndarray (natom, nrad, nang, 3+natom)
            3 for xyz coordinate, last natom cols are atomic weights on this grid point.

        Notes
        -----
        When this method is called, attributes `self.z/x/r/wcheb`, `self.xleb`, and `self.wleb` will be initialized.  
        `self.xleb` (3, nang) is lebedev sampling points coordinates in unit sphere; `self.wleb` is lebedev weight. 
        '''
        self.xleb, self.wleb = lebedev_rule(self.nleb)
        self.zcheb  = np.array(list(range(1, self.ncheb+1)) * self.natom).reshape((self.natom, self.ncheb))
        self.xcheb  = np.zeros((self.natom, self.ncheb))
        self.rcheb  = np.zeros((self.natom, self.ncheb))
        self.wcheb  = np.zeros((self.natom, self.ncheb))
        self.weight = np.zeros((self.natom, self.ncheb*len(self.wleb)))
        grid        = np.zeros((self.natom, self.ncheb, len(self.wleb), 3+self.natom)) # (natom, nrad, nang, 3+natom)
        grid_global = np.zeros((self.natom, self.ncheb*len(self.wleb), 3)) # (natom, nradxnang, 3)

        for iatom, xyz in enumerate(self.xyz):
            xchebs, rchebs, wchebs = gaussCheby2(self.ncheb, self.rms[iatom])
            self.xcheb[iatom] = xchebs
            self.rcheb[iatom] = rchebs
            self.wcheb[iatom] = wchebs
            for j, rcheb in enumerate(rchebs):
                grid[iatom,j,:,:3] = (self.xleb * rcheb + xyz[:,None]).T
                grid[iatom,j,:,3:] = self.get_weight_p(grid[iatom,j,:,:3]).T
            self.weight[iatom] = grid[iatom,:,:,3+iatom].ravel()
            grid_global[iatom] = np.stack((grid[iatom,:,:,0].ravel(), grid[iatom,:,:,1].ravel(), grid[iatom,:,:,2].ravel()), axis=1)

        return grid, grid_global

                

def poisson_solver(rho_lm:np.ndarray, l:int, rcheb:np.ndarray, rm:float=1., qn:float=1.) -> np.ndarray:
    """有限差分求解电子泊松方程计算电子库伦势

    Reference
    ---------
    A. D. Becke; R. M. Dickson J. Chem. Phys. 89, 2993-2997 (1988)
    """
    ncheb = len(rho_lm)

    # for val in rcheb:
    #     print(f"{val:>20.15f}")

    dzdr2  = (ncheb + 1)**2 * rm / (np.pi**2 * rcheb) / (rm + rcheb)**2
    d2zdr2 = (ncheb + 1) * rm**2 * (3 * rcheb + rm) / (2 * np.pi) / (rcheb * rm)**1.5 / (rcheb + rm)**2

    # for val in rcheb:
    #     print(f"{(ncheb + 1) * rm**2 * (3 * val + rm) / (2 * np.pi) / (val * rm)**1.5 / (val + rm)**2:20.15f}")

    b = np.zeros((ncheb+2, ))
    diag = np.diag(l * (l + 1) / rcheb**2)

    # print("#diag")
    # for val in diag.ravel():
    #     print(f"{val:>20.15f}")
    
    # print('#dzdr2')
    # for val in dzdr2:
    #     print(f"{val:>20.15f}")
    # print('#d2zdr2')
    # for val in d2zdr2:
    #     print(f"{val:>20.15f}")


    D1   = np.zeros((ncheb+2, ncheb+2)) # first-order  derivative operator matrix
    D2   = np.zeros((ncheb+2, ncheb+2)) # second-order derivative operator matrix
 
    D1[      1, 0:5] = np.array([-3, -10,  18, -6,   1       ]) / 12
    D1[      2, 0:6] = np.array([ 3, -30, -20, 60, -15,  2   ]) / 60
    D1[ncheb-1, -6:] = np.array([-2,  15, -60, 20,  30, -3   ]) / 60
    D1[  ncheb, -5:] = np.array([-1,   6, -18, 10,   3       ]) / 12
    kernel_7d1       = np.array([-1,   9, -45, 0.,  45, -9, 1]) / 60
    for i in range(3, ncheb-1):
        D1[i, i-3:i+4] = kernel_7d1
    D1[1:-1] *= d2zdr2[:,None]

    D2[      1, 0:5] = np.array([11, -20,   6,    4,  -1        ]) / 12
    D2[      2, 0:5] = np.array([-1,  16, -30,   16,  -1        ]) / 12
    D2[ncheb-1, -5:] = np.array([-1,  16, -30,   16,  -1        ]) / 12
    D2[  ncheb, -5:] = np.array([-1,   4,   6,  -20,  11        ]) / 12
    kernel_7d2       = np.array([ 2, -27, 270, -490, 270, -27, 2]) / 180
    for i in range(3, ncheb-1):
        D2[i, i-3:i+4] = kernel_7d2
    D2[1:-1] *= dzdr2[:,None]

    A = D2 + D1
    A[1:-1,1:-1] -= diag
    A[ 0, 0] = 1.
    A[-1,-1] = 1.
    b[-1] = 0.
    b[0] = np.sqrt(4 * np.pi) * qn if l == 0 else 0.
    b[1:-1] = -4 * np.pi * rcheb * rho_lm
    
    # print('#A')
    # for val in A.ravel():
    #     print(f"{val:>20.15f}")
    # print('#b')
    # for val in b:
    #     print(f"{val:>20.15f}")

    u_lm = np.linalg.solve(A, b)

    # print('#u_lm')
    # for val in u_lm:
    #     print(f"{val:>20.15f}")

    return u_lm



if __name__ == '__main__':
    x, r, w = gaussCheby2(75, rm=0.876)
    # for val in x:
    #     print(f"{val:>20.15f}")
    # print()
    # for val in r:
    #     print(f"{val:>20.15f}")
    # print()
    # for val in w:
    #     print(f"{val:>20.15f}")
    
    l = 4
    _ = poisson_solver(np.arange(1., 76.)/100., l, r, rm=0.876, qn=5.6)
