# -*- coding: utf-8 -*-

import numpy as np
from warnings            import warn
from collections         import Counter
from scipy.linalg        import block_diag
from scipy.integrate     import simpson
from scipy.special       import erfc, erf, spherical_jn
from scipy.interpolate   import CubicSpline
from scipy.sparse.linalg import LinearOperator, lobpcg, eigsh
from upf_tools           import UPFDict
from ase.io              import read


import pylibxc
pbe_x = pylibxc.LibXCFunctional('gga_x_pbe', 'unpolarized')
pbe_c = pylibxc.LibXCFunctional('gga_c_pbe', 'unpolarized')
pz_x = pylibxc.LibXCFunctional('lda_x', 'unpolarized')
pz_c = pylibxc.LibXCFunctional('lda_c_pz', 'unpolarized')


sym2num = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5,
    "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, 'Fe': 26,
}


num2sym = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B",
    6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne"
}


# 2 3 5 #7
# 90: [1,2,1,0] means 90 = 2^1 * 3^2 * 5^1
FFT_GOOD_NUMBER = {
1  : [0, 0, 0, 0], 2  : [1, 0, 0, 0], 3  : [0, 1, 0, 0], 4  : [2, 0, 0, 0], 5  : [0, 0, 1, 0],
6  : [1, 1, 0, 0], 8  : [3, 0, 0, 0], 9  : [0, 2, 0, 0], 10 : [1, 0, 1, 0], 12 : [2, 1, 0, 0],
15 : [0, 1, 1, 0], 16 : [4, 0, 0, 0], 18 : [1, 2, 0, 0], 20 : [2, 0, 1, 0], 24 : [3, 1, 0, 0],
25 : [0, 0, 2, 0], 27 : [0, 3, 0, 0], 30 : [1, 1, 1, 0], 32 : [5, 0, 0, 0], 36 : [2, 2, 0, 0],
40 : [3, 0, 1, 0], 45 : [0, 2, 1, 0], 48 : [4, 1, 0, 0], 50 : [1, 0, 2, 0], 54 : [1, 3, 0, 0],
60 : [2, 1, 1, 0], 64 : [6, 0, 0, 0], 72 : [3, 2, 0, 0], 75 : [0, 1, 2, 0], 80 : [4, 0, 1, 0],
81 : [0, 4, 0, 0], 90 : [1, 2, 1, 0], 96 : [5, 1, 0, 0], 100: [2, 0, 2, 0], 108: [2, 3, 0, 0],
120: [3, 1, 1, 0], 125: [0, 0, 3, 0], 128: [7, 0, 0, 0], 135: [0, 3, 1, 0], 144: [4, 2, 0, 0],
150: [1, 1, 2, 0], 160: [5, 0, 1, 0], 162: [1, 4, 0, 0], 180: [2, 2, 1, 0], 192: [6, 1, 0, 0],
200: [3, 0, 2, 0],
}


def gram_schmidt(nrow:int, ncol:int):
    # qr decomposition seems to be more stable
    uG = np.random.randn(nrow, ncol) + 1j * np.random.randn(nrow, ncol)
    uG[:,0] /= np.sqrt(np.real(np.sum((np.conjugate(uG[:,0]) * uG[:,0]))))
    
    for i in range(1, ncol):
        v = uG[:,i].copy()
        for j in range(i):
            v -= np.sum(np.conjugate(uG[:,j]) * uG[:,i]) * uG[:,j]
        v /= np.sqrt(np.real(np.sum((np.conjugate(v) * v))))
        uG[:,i] = v
    
    return uG


def real_spherical_harmonics(l:int, m:int, xyz:np.ndarray) -> np.ndarray:
    # xyz : np.ndarray (n, 3) make sure these vectors are normalized!
    assert l >= 0 and -l <= m <= l
    ylm = np.zeros((len(xyz)), dtype=float)

    if l == 0:
        ylm += np.sqrt(1 / 4 / np.pi)
        return ylm

    if l == 1:
        if   m == -1: ylm = np.sqrt(3 / 4 / np.pi) * xyz[:,1]
        elif m ==  0: ylm = np.sqrt(3 / 4 / np.pi) * xyz[:,2]
        elif m ==  1: ylm = np.sqrt(3 / 4 / np.pi) * xyz[:,0]
    elif l == 2:
        if   m == -2: ylm = np.sqrt(15 / 4 / np.pi) * (xyz[:,0] * xyz[:,1])
        elif m == -1: ylm = np.sqrt(15 / 4 / np.pi) * (xyz[:,1] * xyz[:,2])
        elif m ==  0: ylm = np.sqrt(5 / 16 / np.pi) * (2 * xyz[:,2]**2 - xyz[:,0]**2 - xyz[:,1]**2)
        elif m ==  1: ylm = np.sqrt(15 / 4 / np.pi) * (xyz[:,0] * xyz[:,2])
        elif m ==  2: ylm = np.sqrt(15 / 16 / np.pi) * (xyz[:,0]**2 - xyz[:,1]**2)
    elif l == 3:
        if   m == -3: ylm = np.sqrt( 35 / 32 / np.pi) * (3 * xyz[:,0]**2 - xyz[:,1]**2) * xyz[:,1]
        elif m == -2: ylm = np.sqrt(105 /  4 / np.pi) * xyz[:,0] * xyz[:,1] * xyz[:,2]
        elif m == -1: ylm = np.sqrt( 21 / 32 / np.pi) * xyz[:,1] * (4 * xyz[:,2]**2 -     xyz[:,0]**2     - xyz[:,1]**2)
        elif m ==  0: ylm = np.sqrt(  7 / 16 / np.pi) * xyz[:,2] * (2 * xyz[:,2]**2 - 3 * xyz[:,0]**2 - 3 * xyz[:,1]**2)
        elif m ==  1: ylm = np.sqrt( 21 / 32 / np.pi) * xyz[:,0] * (4 * xyz[:,2]**2 -     xyz[:,0]**2     - xyz[:,1]**2)
        elif m ==  2: ylm = np.sqrt(105 / 16 / np.pi) * (xyz[:,0]**2 - xyz[:,1]**2) * xyz[:,2]
        elif m ==  3: ylm = np.sqrt( 35 / 32 / np.pi) * (xyz[:,0]**2 - 3 * xyz[:,1]**2) * xyz[:,0]
    else:
        raise NotImplementedError()

    return ylm


r'''
orthorhombic box only, closed-shell and spin-unpolarized system only, k=(0,0,0) only but use complex wave function

.cif file uses Angstrom, converted to Bohr 

.upf file uses Ry, converted to Hartree

radial grid in sg15 ONCV is uniform

plane wave basis convention: |G> = 1/\sqrt{V} e^{i G \cdot r}

numpy fft normalization convention used

combined with plane wave normalization convention, extra normalization factor \sqrt{V}/N1N2N3 for fft and N1N2N3/\sqrt{V} for ifft

projector radial function \beta is stored as r\beta in .upf

RHOATOM in .upf is stored as 4\pi r^2 \rho

sg15 ONCV uses PBE
'''
class SuperCell:
    def __init__(self, cif:str, ecutwfc:float, ppdir:str=None, ecutrho:float=None):
        self.cif       : str        = cif     # cif file path
        self.ppdir     : str        = ppdir   # upf files path
        self.cell_real : np.ndarray = None    # row is lattice vector!
        self.cell_reci : np.ndarray = None
        self.V         : float      = None    # cell volume
        self.dV        : float      = None
        self.N3        : float      = None    # nx * ny * nz, for normalization convention
        self.xyz_cart  : np.ndarray = None    # Cartesian coord, sorted by self.atmlist afterwards
        self.ntyp      : int        = None    # number of types of elements
        self.natm      : int        = None    # number of total atoms
        self.numbers   : list       = None    # atomic numbers,            i.e. C3H6O -> H6C3O -> [1,1,1,1,1,1,6,6,6,8]
        self.pp_numbers : list      = None    # pseudized nuclear charges, i.e. H6C3O -> [1,1,1,1,1,1,4,4,4,6]
        self.vellist   : list       = []      # grouped valence charges,   i.e. C3H6O -> HCO   -> [1, 4, 6]
        self.symlist   : list       = []      # grouped element symbols,   i.e. C3H60 -> HCO   -> [H, C, O]
        self.numlist   : list       = []      # grouped atomic numbers,    i.e. C3H6O -> HCO   -> [1, 6, 8]
        self.atmlist   : list       = []      # grouped atom counters,     i.e. C3H6O -> H6C3O -> [6, 3, 1]
        self.nve       : int        = None    # total number of valence electron, closed-shell only!
        self.nbnd      : int        = None    # number of filled bands, closed-shell only!
        self.ewald     : float      = None    # Ewald energy contribution
        self.ecutwfc   : float      = ecutwfc # wavefunction kinetic energy cutoff
        self.ecutrho   : float      = ecutrho # should be 4 * ecutwfc
        self.nx        : int        = None    # discretization, controls fft grid size and number of G-vectors
        self.ny        : int        = None
        self.nz        : int        = None
        self.nGvec     : int        = None    # number of G-vec, fulfilling |G|^2/2 <= Ecutrho
        self.Gvec      : np.ndarray = None    # (nGvec, 3) G-vectors within cutoff
        self.G2        : np.ndarray = None    # G-vector norm square, (nGvec,)
        self.Gnorm     : np.ndarray = None   # G-vector norm, (nGvec,)
        self.strfcs    : np.ndarray = None    # structure factors e^{-i G \cdot r_j}, (ntyp, nGvec) sum out all atm of the same species
        self.ylm       : np.ndarray = None    # ((lmax+1)^2, nGvec), real spherical harmonics of normalized G-vec
        self.id_cut    : tuple      = None    # ((idx), (idy), (idz)) mask indices tuple

        # pseudo potential artributes, grouped on elements
        self.pp_r      : list[np.ndarray] = [] # grouped r, uniform grid only!
        self.pp_rab    : list[np.ndarray] = [] # grouped rab
        self.pp_vloc   : list[np.ndarray] = [] # grouped vloc
        self.pp_rho    : list[np.ndarray] = [] # grouped ve density, can serve as initial guess
        self.pp_beta   : list[np.ndarray] = [] # grouped vnl radial projectors β (nproj, nmesh) 2D
        self.pp_l      : list[np.ndarray] = [] # grouped vnl l-channel (lmax+1,) 1D
        self.pp_nid    : list[np.ndarray] = [] # grouped vnl number of projs in one l-channel (lmax+1,) 1D
        self.pp_rcid   : list[np.ndarray] = [] # grouped vnl beta cutoff radius index <(nproj,) 1D
        self.pp_dij    : list[np.ndarray] = [] # grouped coupling matrix for ∑_lm ∑_ij |βlmi> D_{l,ij} <βlmj|


    def init(self, psp_dict:dict, verbose:bool=False):
        '''
        Example
        -------
        >>> psp_dict = {'C': 'C_ONCV_PBE-1.2.upf', 'N': 'N_ONCV_PBE-1.2.upf'}
        >>> cell.init(psp_dict)
        '''

        # read mol info
        mol            = read(self.cif, index=0)
        self.cell_real = np.array(list(mol.cell)) / 0.529177210903 # angstrom to bohr, cif file in angstrom!
        self.cell_reci = np.linalg.inv(self.cell_real) * 2 * np.pi
        self.V         = abs(np.linalg.det(self.cell_real))
        mask           = np.argsort(mol.get_atomic_numbers(), kind='stable')
        self.numbers   = mol.get_atomic_numbers()[mask].tolist() # ascending
        self.xyz_cart  = mol.positions[mask] / 0.529177210903 # angstrom to bohr
        symbols        = np.array(mol.get_chemical_symbols())[mask] # ascending in atomic numbers
        self.natm      = len(symbols)
        counter        = Counter(symbols)
        self.ntyp      = len(counter)                             # unique types
        self.atmlist   = [int(val) for val in counter.values()]   # occurance of element, atomic number
        self.symlist   = [str(key) for key in counter.keys()]     # occurance of element, element symbol
        self.numlist   = [sym2num[key] for key in counter.keys()] # which element

        # read upf and extract vloc, vnl, rho, r, rab
        if self.ppdir is None:
            self.ppdir = './'
        if self.ppdir[-1] != '/':
            self.ppdir += '/'
        for sym in self.symlist: # loop over all unique element, atomic number ascending order
            if sym not in psp_dict.keys():
                raise ValueError(f'{sym} pseudo potential file not found')
            ppfile = self.ppdir + psp_dict[sym] # i.e. '../ppdir/' + 'X_ONCV_PBE-1.2.upf'
            pp = UPFDict.from_upf(ppfile)

            # r, rab, Vloc, RHOATOM
            self.pp_rho.append(pp['rhoatom'])
            self.pp_r.append(pp['mesh']['r']) # in bohr
            self.pp_rab.append(pp['mesh']['rab'])
            self.pp_vloc.append(pp['local'] / 2.) # Ry to Ht
            self.vellist.append(int(pp['header']['z_valence']))

            # non-local part
            # one element can have multiple l-channels: s, p, d, f, ...
            # and one l-channel can have multiple projectors, i, j, ...
            # |β> D <β| in Ry/Bohr unit
            # l ascending order
            nproj = pp['header']['number_of_proj']
            nmesh = pp['header']['mesh_size']
            dij   = pp['nonlocal']['dij'].reshape((nproj, nproj))
            rcid  = [beta['cutoff_radius_index']-1 for beta in pp['nonlocal']['beta']] # upf is 1-based
            projs = np.zeros((nproj, nmesh), dtype=float) # (nproj, nmesh) projectors 2D mat
            for ibeta, beta in enumerate(pp['nonlocal']['beta']):
                projs[ibeta] = beta['content']

            # nid bookkeep l, ij in l, and corresponding dij
            angls = [beta['angular_momentum'] for beta in pp['nonlocal']['beta']] # i.e. [0,0,1,1,2,2,3,3] if lmax=3, ungrouped
            counter = Counter(angls)
            pp_l = [key for key in counter.keys()] # [0,1,2,3] if lmax=3, grouped
            nid = [val for val in counter.values()] # [2,2,2,2] means 2 for l=0, 2 for l=1, ... how many projs in one l-channel
            self.pp_l.append(pp_l)
            self.pp_rcid.append(rcid)
            self.pp_beta.append(projs)
            self.pp_dij.append(dij)
            self.pp_nid.append(nid)
        
        self.pp_numbers = []
        for vel, n in zip(self.vellist, self.atmlist):
            self.pp_numbers.extend([vel] * n) # ungroup

        self.nve = int(np.sum(self.pp_numbers))
        self.nbnd = self.nve // 2
        self._build_3D_grid() # nGvec, Gvec, G2, Gnorm, id_cut
        self.ylm             = self._compute_ylm()
        
        # self.strfcs = np.exp(-1j * (self.xyz_cart @ self.Gvec.T)) # (natm, nGvec)
        self.strfcs = np.zeros((self.ntyp, self.nGvec), dtype=complex)
        for ityp in range(self.ntyp):
            ub = np.sum(self.atmlist[:ityp+1])
            lb = ub - self.atmlist[ityp]
            self.strfcs[ityp] = np.sum(np.exp(-1j * (self.xyz_cart[lb:ub] @ self.Gvec.T)), axis=0) # (ntyp, nGvec)

        self.ewald           = self._compute_ewald()

        if verbose:
            self._logger()


    def _build_3D_grid(self):
        L1, L2, L3 = np.diagonal(self.cell_real)
        if self.ecutrho is None or abs(4 * self.ecutwfc - self.ecutrho) <= 1e-16:
            self.ecutrho = 4 * self.ecutwfc
            kmax_fft_x = int(np.floor(np.sqrt(2 * self.ecutrho) * L1 / 2 / np.pi))
            kmax_fft_y = int(np.floor(np.sqrt(2 * self.ecutrho) * L2 / 2 / np.pi))
            kmax_fft_z = int(np.floor(np.sqrt(2 * self.ecutrho) * L3 / 2 / np.pi))
            nx = 2 * kmax_fft_x + 1
            ny = 2 * kmax_fft_y + 1
            nz = 2 * kmax_fft_z + 1

            while nx not in FFT_GOOD_NUMBER.keys(): nx += 1
            while ny not in FFT_GOOD_NUMBER.keys(): ny += 1
            while nz not in FFT_GOOD_NUMBER.keys(): nz += 1

            k_fft_x = np.fft.fftfreq(nx, 1./nx).astype(int) # [0,1,2,...,nfft/2, -nfft/2,...,-1]
            k_fft_y = np.fft.fftfreq(ny, 1./ny).astype(int)
            k_fft_z = np.fft.fftfreq(nz, 1./nz).astype(int)
            kx, ky, kz = np.meshgrid(k_fft_x, k_fft_y, k_fft_z, indexing='ij')
            dV = (L1 / nx) * (L2 / ny) * (L3 / nz)
            N3 = nx * ny * nz
            
            G2 = (kx / L1 * 2 * np.pi)**2 + (ky / L2 * 2 * np.pi)**2 + (kz / L3 * 2 * np.pi)**2
            mask = G2 <= 2 * self.ecutrho + 1e-12
            idx = (kx[mask] + nx) % nx
            idy = (ky[mask] + ny) % ny
            idz = (kz[mask] + nz) % nz

            nGvec      = len(idx)
            id_cut     = (idx, idy, idz)
            G2      = G2[mask]
            Gnorm = np.sqrt(G2)
            Gvec       = np.zeros((nGvec, 3), dtype=float)
            Gvec[:,0]  = kx[id_cut] / L1 * 2 * np.pi
            Gvec[:,1]  = ky[id_cut] / L2 * 2 * np.pi
            Gvec[:,2]  = kz[id_cut] / L3 * 2 * np.pi

            assert 0 <= G2[0] <= 1e-16
            assert 0 <= Gvec[0,0] <= 1e-16 and 0 <= Gvec[0,1] <= 1e-16 and 0 <= Gvec[0,2] <= 1e-16
            assert np.all(G2[1:] > 1e-16)

            self.nGvec  = nGvec
            self.nx     = nx
            self.ny     = ny
            self.nz     = nz
            self.dV     = dV
            self.N3     = N3
            self.id_cut = id_cut
            self.G2     = G2
            self.Gnorm  = Gnorm
            self.Gvec   = Gvec

        elif self.ecutrho < 4 * self.ecutwfc:
            raise ValueError("ecutrho >= 4 * ecutwfc")
        else:
            raise NotImplementedError("ecutrho > 4 * ecutwfc will make two sets of fft grids and G-vectors, which is not yet available")


    def _compute_ylm(self) -> np.ndarray:
        '''precompute real spherical harmonics for all G-vec up to lmax'''
        # |G|=0 skipped
        Gvec_normalized = self.Gvec[1:] / np.linalg.norm(self.Gvec[1:], axis=1)[:,None] # (nGvec-1,)
        
        lmax = 0
        for _lmax in self.pp_l:
            if _lmax[-1] > lmax:
                lmax = _lmax[-1]
        
        ylm = np.zeros(((lmax+1)**2, self.nGvec), dtype=float)
        i = 0
        for l in range(lmax+1):
            for m in range(-l, l+1):
                ylm[i,1:] = real_spherical_harmonics(l, m, Gvec_normalized) # skip |G|=0, ylm[:,0] = 0, use with caution!
                i += 1

        return ylm


    def _compute_ewald(self) -> float:
        '''Ewald summation'''
        Q = np.sum(self.pp_numbers)
        ecut = 3. # Ht
        Gmax = np.sqrt(ecut * 2) # |G|^2/2 <= ecut

        # iteratively find optimum splitting param alpha
        alpha = 3.
        etail = 42.
        while etail > 1e-7:
            alpha -= 0.05
            if alpha < 0:
                raise ValueError("Failed to find optimum alpha")
            etail = 2 * Q**2 * np.sqrt(alpha / np.pi) * erfc(np.sqrt(ecut * 2 / 4 / alpha))

        # short range energy
        rmax = 4 / np.sqrt(alpha)
        Nrx, Nry, Nrz = np.ceil(rmax / np.diagonal(self.cell_real)).astype(int) + 2 # 2 here is buffer
        _Nrx = np.arange(-Nrx, Nrx+1)
        _Nry = np.arange(-Nry, Nry+1)
        _Nrz = np.arange(-Nrz, Nrz+1)
        M = np.transpose(np.meshgrid(_Nrx, _Nry, _Nrz, indexing='ij')).reshape((-1, 3)) # a trick equivalent to X,Y,Z = meshgrid... stack(X.ravel(), ...)
        mask = np.any(M, axis=1) # kick out [0,0,0]
        M = M[mask] @ self.cell_real # lattice vector n = Nrx a1 + Nry a2 + Nrz a3

        ewald_sr = 0.
        for i, (ixyz, iq) in enumerate(zip(self.xyz_cart, self.pp_numbers)):
            for j, (jxyz, jq) in enumerate(zip(self.xyz_cart, self.pp_numbers)):
                for n in M: # loop over lattice vector n
                    _r = np.linalg.norm(ixyz - jxyz + n)
                    ewald_sr += 1 / 2 * iq * jq / _r * erfc(np.sqrt(alpha) * _r)
                if i != j: # different ions in the original cell
                    _r = np.linalg.norm(ixyz - jxyz)
                    ewald_sr += 1 / 2 * iq * jq / _r * erfc(np.sqrt(alpha) * _r)

        # long range energy
        bs = 2 * np.pi / np.diagonal(self.cell_real)
        Ngx, Ngy, Ngz = np.ceil(Gmax / bs).astype(int) + 2
        _Ngx = np.arange(-Ngx, Ngx+1)
        _Ngy = np.arange(-Ngy, Ngy+1)
        _Ngz = np.arange(-Ngz, Ngz+1)
        M = np.transpose(np.meshgrid(_Ngx, _Ngy, _Ngz, indexing='ij')).reshape((-1, 3))
        mask = np.any(M, axis=1)
        M = M[mask] @ self.cell_reci

        ewald_lr = 0.
        for n in M:
            strf = 0. # structure factor e^{-i G \cdot R_j}
            for i, (ixyz, iq) in enumerate(zip(self.xyz_cart, self.pp_numbers)):
                strf += iq * np.exp(-1j * np.dot(n, ixyz)) # q_i exp[-I G r_i]
            strf = np.abs(strf)**2
            G2 = np.dot(n, n)
            ewald_lr += 2 * np.pi / self.V * np.exp(-G2 / 4 / alpha) / G2 * strf

        # self interaction and G=0 term correction
        q2total = 0
        for iq in self.pp_numbers:
            q2total += iq**2
        ewald_self = np.sqrt(alpha / np.pi) * q2total
        ewald_G0 = np.pi / 2 / alpha / self.V * Q**2

        return ewald_lr + ewald_sr - ewald_self - ewald_G0


    def _logger(self):
        print('===== system info =====')
        print(f"L1, L2, L3    = {self.cell_real[0,0]}, {self.cell_real[1,1]}, {self.cell_real[2,2]} a.u.")
        print(f"natom         = {self.natm}")
        print(f"ntype         = {self.ntyp}")
        print(f"symbol list   = {self.symlist}")
        print(f"vellist       = {self.vellist}")
        print(f"number list   = {self.numlist}")
        print(f"atom list     = {self.atmlist}")
        print(f"pp nuc charge = {self.pp_numbers}")
        print(f"nuc charge    = {self.numbers}")
        print()
        print(f"ntype pp mesh = {len(self.pp_r)} and length for each {[len(mesh) for mesh in self.pp_r]}")
        print(f"ntype pp Vloc = {len(self.pp_vloc)}")
        print(f"pp l-channel  = {self.pp_l}")
        print(f"pp nid        = {self.pp_nid}")
        print(f"pp beta       = {[beta.shape for beta in self.pp_beta]}")
        print()
        print(f"ecutwfc = {self.ecutwfc} Ht    ecutrho = {self.ecutrho} Ht")
        print(f"Ewald contribution = {self.ewald*2} Ry")
        print(f"fft grid shape     = {(self.nx, self.ny, self.nz)}")
        print(f"number G-vectors   = {self.nGvec}")
        print(f"structure factors shape = {self.strfcs.shape}")
        print('=======================\n')


    def compute_eden_init_real(self, do_interpolation:bool=True, renomalize:bool=True) -> np.ndarray:
        '''build superposition of atomic density from RHOATOM as initial guess'''

        eden_fourier = np.zeros((self.nGvec,), dtype=complex)
        eden_init    = np.zeros((self.nx, self.ny, self.nz), dtype=float)

        for ityp in range(self.ntyp):
            r             = self.pp_r[ityp]
            rab           = self.pp_rab[ityp]
            rho           = self.pp_rho[ityp] # (len(r),), radial density, stored as 4 π r^2 ρ(r) in upf file
            if do_interpolation:
                G2max = self.ecutrho * 2
                G2min   = 0.
                G_table = np.linspace(np.sqrt(G2min), np.sqrt(G2max), 4096, endpoint=True)
                Gr = np.outer(G_table, r)
                j0 = np.sinc(Gr / np.pi)
                interp_table = 1 / np.sqrt(self.V) * simpson(rho * j0, x=r, axis=1)
                interpolator = CubicSpline(G_table, interp_table, extrapolate=False)
                radial_int = interpolator(self.Gnorm)
                eden_fourier += self.strfcs[ityp] * radial_int
            else:
                Gr            = np.outer(self.Gnorm, r) # (nGvec, len(r))
                j0            = np.sinc(Gr / np.pi)
                radial_int    = 1 / np.sqrt(self.V) * simpson(rho * j0, x=r, axis=1) # ∫ [(ρ(r) r^2 4 π)/(4 π r^2)] sin(Gr)/(Gr) dr -> (nGvec,)
                eden_fourier += self.strfcs[ityp] * radial_int # (nGvec,)

        fft_grid = np.zeros((self.nx, self.ny, self.nz), dtype=complex)
        fft_grid[self.id_cut] = eden_fourier
        eden_init = self.N3 / np.sqrt(self.V) * np.fft.ifftn(fft_grid).real
        
        if renomalize:
            nve = np.sum(eden_init * self.dV)
            if abs(nve - self.nve) >= 1e-1:
                print(f'Before renormalization nve = {nve}')
            eden_init *= self.nve / nve # renormalize
            
        return eden_init


    def compute_eden_real(self, uG:np.ndarray) -> np.ndarray:
        '''compute ρ(r) on 3D real mesh

        Arguments
        ---------
        uG : np.ndarray
            (nGvec, nbnd) wave function expansion coefficient matrix

        return
        ------
        eden : np.ndarray
            (nx, ny, nz) valence electron density on 3D mesh
        '''

        _, nbnd = uG.shape
        fft_batch_grid = np.zeros((nbnd, self.nx, self.ny, self.nz), dtype=complex)
        fft_batch_grid[:, self.id_cut[0], self.id_cut[1], self.id_cut[2]] = uG.T
        tmp_batch = np.fft.ifftn(fft_batch_grid, axes=(1,2,3)) * self.N3 / np.sqrt(self.V)
        eden = np.sum((2. * tmp_batch * tmp_batch.conj()).real, axis=0)

        return eden


    def compute_vHa_real(self, eden:np.ndarray) -> tuple[np.ndarray, float]:
        '''compute Hartree potential vHa(r) on 3D real mesh

        Arguments
        ---------
        eden : np.ndarray
            (nx, ny, nz) valence electron density on 3D real mesh

        return
        ------
        (vHa, eHa) : [np.ndarray, float]
            (nx, ny, nz) Hartree potential on 3D real mesh, force to be real, and Hartree energy
        '''

        G2 = self.G2.copy()
        G2[0] = 1.
        fft_grid = np.zeros((self.nx, self.ny, self.nz), dtype=complex)
        eden_fourier = np.fft.fftn(eden)[self.id_cut] # (nx, ny, nz) -> (nGvec,)
        fft_grid[self.id_cut] = 4 * np.pi / G2 * eden_fourier
        fft_grid[0,0,0] = 0.+0j
        vHa = np.fft.ifftn(fft_grid).real # (nx, ny, nz)
        eHa = np.sum(vHa * eden * self.dV) / 2.
        
        return vHa, eHa


    def compute_vxc_real_pbe(self, eden:np.ndarray) -> tuple[np.ndarray, float]:
        '''
        GGA Vxc on 3D real grid

        Arguments
        ---------
        eden : np.ndarray
            (nx, ny, nz) valence electron denity on 3D real grid. Make sure it's non-negative!
        
        return
        ------
            (vxc, exc) : [np.ndarray, float]
                (nx, ny, nz) Vxc on 3D real grid, and energy
        '''

        # prepare rho and sigma
        fft_grid = np.zeros((self.nx, self.ny, self.nz), dtype=complex)
        eden_fourier = np.fft.fftn(eden)[self.id_cut] # (nGvec,)
        eden_grad = [np.zeros((self.nx, self.ny, self.nz), dtype=float), 
                     np.zeros((self.nx, self.ny, self.nz), dtype=float), 
                     np.zeros((self.nx, self.ny, self.nz), dtype=float), ] # [∂xρ, ∂yρ, ∂zρ]
        sigma = np.zeros((self.nx, self.ny, self.nz), dtype=float) # |∇ρ|^2
        
        for i in range(3):
            fft_grid[self.id_cut] = 1j * self.Gvec[:,i] * eden_fourier
            eden_grad[i] = np.fft.ifftn(fft_grid).real
            sigma += eden_grad[i]**2
        
        # libxc
        inp = {'rho': eden, 'sigma': sigma}
        ret_x  = pbe_x.compute(inp)
        ret_c  = pbe_c.compute(inp)
        vrho   = ret_x['vrho'].ravel().reshape(eden.shape)   + ret_c['vrho'].ravel().reshape(eden.shape)
        vsigma = ret_x['vsigma'].ravel().reshape(eden.shape) + ret_c['vsigma'].ravel().reshape(eden.shape)
        zk     = ret_x['zk'].ravel().reshape(eden.shape)     + ret_c['zk'].ravel().reshape(eden.shape)

        # build vxc and exc
        vsigma_grad_eden_fourier = [np.fft.fftn(vsigma * eden_grad[0])[self.id_cut], 
                                    np.fft.fftn(vsigma * eden_grad[1])[self.id_cut], 
                                    np.fft.fftn(vsigma * eden_grad[2])[self.id_cut], ] # F[Vσ⋅∇ρ] = [(nGvec,), (nGvec,), (nGvec,)]
        
        fft_grid.fill(0.+0j)
        for i in range(3):
            fft_grid[self.id_cut] += 1j * self.Gvec[:,i] * vsigma_grad_eden_fourier[i]
        div = np.fft.ifftn(fft_grid).real

        vxc = vrho - 2 * div # Vρ - 2 ∇ ⋅ (Vσ ∇ρ)
        exc = np.sum(zk * eden * self.dV)

        return vxc, exc


    def compute_vxc_real_pz(self, eden:np.ndarray) -> tuple[np.ndarray, float]:
        '''
        PZ vxc on 3D real grid

        Arguments
        ---------
        eden : np.ndarray
            (nx, ny, nz) valence electron denity on 3D real grid. Make sure it's non-negative!
        
        return
        ------
            (vxc, exc) : [np.ndarray, float]
                (nx, ny, nz) Vxc on 3D real grid, and energy
        '''
        
        # libxc
        inp = {'rho': eden}
        ret_x = pz_x.compute(inp)
        ret_c = pz_c.compute(inp)
        vrho  = ret_x['vrho'].ravel().reshape(eden.shape) + ret_c['vrho'].ravel().reshape(eden.shape)
        zk    = ret_x['zk'].ravel().reshape(eden.shape)   + ret_c['zk'].ravel().reshape(eden.shape)
        exc   = np.sum(zk * eden * self.dV)

        return vrho, exc


    def compute_vloc_fourier(self, do_interpolation:bool=True) -> np.ndarray:
        '''
        Return
        ------
        vloc_fourier : np.ndarray
            (nGvec,), use loop to loop over bands
        '''

        vloc_fourier = np.zeros((self.nGvec,), dtype=complex) # (nGvec,) <G|Vloc>
        vloc_G0 = 0.

        for ityp in range(self.ntyp):
            r      = self.pp_r[ityp]
            Za     = self.vellist[ityp]
            vloc   = self.pp_vloc[ityp]
            kernel = 4 * np.pi * r * (r * vloc + Za * erf(r)) # (len(r),)
            lr = np.zeros((self.nGvec,), dtype=float)
            lr[1:] = -Za * 4 * np.pi / self.G2[1:] * np.exp(-self.G2[1:] / 4.) # (nGvec,)

            if do_interpolation:
                G2max = self.ecutrho * 2
                G2min = 0.
                G_table = np.linspace(np.sqrt(G2min), np.sqrt(G2max), 4096, endpoint=True)
                Gr = np.outer(G_table, r)
                j0 = np.sinc(Gr / np.pi)
                interp_table = simpson(kernel * j0, x=r, axis=1)
                interpolator = CubicSpline(G_table, interp_table, extrapolate=False)
                sr = interpolator(self.Gnorm) # (nGvec)
            else:
                Gr = np.outer(self.Gnorm, r) # (nGvec, len(r))
                j0 = np.sinc(Gr / np.pi)
                sr = simpson(kernel * j0, x=r, axis=1) # ∫ {4 π (vloc - -Z erf(r)/r) r^2} sin(Gr)/(Gr) dr -> (nGvec,)

            vloc_G0 += self.atmlist[ityp] * simpson(4 * np.pi * r * (r * vloc + Za), x=r)
            vloc_fourier += 1 / np.sqrt(self.V) * self.strfcs[ityp] * (sr + lr) # (nGvec,)
        
        vloc_fourier[0] = vloc_G0 / np.sqrt(self.V)

        return vloc_fourier
    

    def compute_vloc_real(self) -> np.ndarray:
        fft_grid = np.zeros((self.nx, self.ny, self.nz), dtype=complex)
        fft_grid[self.id_cut] = self.compute_vloc_fourier()
        vloc_real = self.N3 / np.sqrt(self.V) * np.fft.ifftn(fft_grid).real # (nx, ny, nz)

        return vloc_real

    

        '''
        compute <G+k|β> for all typ, only once

        i.e. C has l=0,1, and 2 projectors for each l-channel, considering (-1j)^l Ylm 
        there are [(0,0), (0,0) || (1,-1), (1,-1) || (1,0), (1,0) || (1,1), (1,1)],
        therefore <G+k|β> has shape (nGvec, nproj=8)
        
        return
        ------
        vnl_fourier : list[np.ndarray]
            [ntyp, (nGvec, nprojtot)]. When compute <G+k|β>D<β|ψ> (l,m) pair has to be carefully bookkeeped

            i.e. for C l=0,1, 2 projectors for each l-channel, <G+k|β> is shape (nGvec,8) array:
              p0    p1  |   p0    p1    |   p0    p1  |   p0    p1 
            (0,0) (0,0) | (1,-1) (1,-1) | (1,0) (1,0) | (1,1) (1,1) 
        '''
    def compute_vnl_fourier(self, do_interpolation:bool=True) -> list[np.ndarray]:
        G2max = self.ecutrho * 2 + 1e-12
        G2min = 0.
        Gtable = np.linspace(np.sqrt(G2min), np.sqrt(G2max), 4096, endpoint=True) if do_interpolation else None
        vnl_fourier = [] # [natm, (nGvec, nprojtot)]

        for ityp in range(self.ntyp):
            nlm     = np.sum([(2 * l + 1) * n for l, n in zip(self.pp_l[ityp], self.pp_nid[ityp])])
            ylm_ids = np.concatenate([np.repeat(l**2 + l + np.arange(-l,l+1), n) for l, n in zip(self.pp_l[ityp], self.pp_nid[ityp])])
            phases  = np.concatenate([np.repeat((-1j)**l, (2 * l + 1) * n) for l, n in zip(self.pp_l[ityp], self.pp_nid[ityp])])

            vnl         = np.zeros((self.nGvec, nlm), dtype=complex)
            kernels     = self.pp_beta[ityp] * self.pp_r[ityp]
            corrections = simpson(kernels, x=self.pp_r[ityp], axis=1)

            iproj = 0
            lb    = 0
            for l, n in zip(self.pp_l[ityp], self.pp_nid[ityp]):
                ncol = (2 * l + 1) * n
                ub = lb + ncol
                tmp = np.zeros((self.nGvec, n), dtype=float)

                for i in range(n):
                    rcid = self.pp_rcid[ityp][iproj]
                    rcut = self.pp_r[ityp][:rcid+1]
                    kernel = kernels[iproj][:rcid+1]
                    if do_interpolation:
                        Gr = np.outer(Gtable, rcut)
                        jl = spherical_jn(l, Gr)
                        interp_table = simpson(kernel * jl, x=rcut, axis=1)
                        interpolator = CubicSpline(Gtable, interp_table, extrapolate=False)
                        tmp[:,i] = interpolator(self.Gnorm)
                    else:
                        Gr = np.outer(self.Gnorm, rcut)
                        jl = spherical_jn(l, Gr)
                        tmp[:,i] = simpson(kernel * jl, x=rcut, axis=1)
                    
                    tmp[0,i] = 0. if l > 0 else np.sqrt(4 * np.pi / self.V) * corrections[iproj]
                    iproj += 1
                
                vnl[:,lb:ub] = tmp[:,np.arange(ncol)%n]
                lb = ub
            
            ylm = self.ylm[ylm_ids, 1:].T
            prefactor = 4 * np.pi * phases * ylm / np.sqrt(self.V)
            vnl[1:,:] *= prefactor
            atm_ub = np.sum(self.atmlist[:ityp+1])
            atm_lb = atm_ub - self.atmlist[ityp]
            for ia in range(atm_lb, atm_ub):
                phase = np.exp(-1j * (self.Gvec @ self.xyz_cart[ia]))
                vnl_fourier.append(vnl * phase[:,None])
        
        return vnl_fourier


class LinOpH(LinearOperator):
    def __init__(self, nGvec:int, id_cut:tuple, G2:np.ndarray, Dij:list[np.ndarray], atmlist:list, ecutwfc:float):
        super().__init__(shape=(nGvec, nGvec), dtype=np.dtype(complex))
        # passed once and cached
        self.nGvec  = nGvec
        self.id_cut = id_cut
        self.G2  = G2
        self.Dij = Dij
        self.atmlist = atmlist # track atm per typ
        self.vnl_fourier = None # <G+k|βlmi> [ntyp, (nGvec, nprojtot)]
        self.mask = self.G2 <= 2 * ecutwfc
        
        # passed dynamically
        self.vlocal_real = None # (nx, ny, nz)


    def _matmat(self, uG:np.ndarray):
        '''
        Arguments
        ---------
        uG : np.ndarray
            (nGvec, nbnd), trial wave vectors of all requested bands from last round SCF loop
        '''

        nbnd = uG.shape[1]
        fft_grid = np.zeros(self.vlocal_real.shape, dtype=complex)

        # kinetic
        TPsi_fourier = self.G2[:,None] * uG / 2 # (nGvec, nbnd)

        # pp_loc + vxc + vHa
        vlocalPsi_fourier = np.zeros((self.nGvec, nbnd), dtype=complex) # (nGvec, nbnd)
        for ibnd in range(nbnd):
            fft_grid[self.id_cut] = uG[:,ibnd]
            vlocalPsi_fourier[:,ibnd] = np.fft.fftn(self.vlocal_real * np.fft.ifftn(fft_grid))[self.id_cut]

        # pp_nonlocal
        vnlocalPsi_fourier = np.zeros((self.nGvec, nbnd), dtype=complex) # (nGvec, nbnd)
        iatm = 0
        for ityp, natm in enumerate(self.atmlist):
            for _ in range(natm):
                vnl = self.vnl_fourier[iatm]
                vnlocalPsi_fourier += (vnl @ self.Dij[ityp]) @ (vnl.T.conj() @ uG)
                iatm += 1
        vnlocalPsi_fourier /= 2. # Ry to Ht

        HPsi_fourier = vlocalPsi_fourier + vnlocalPsi_fourier + TPsi_fourier # (nGvec, nbnd)
        HPsi_fourier[~self.mask,:] = 0.+0j

        return HPsi_fourier


    def _matvec(self, uG:np.ndarray):
        uG = uG.ravel()
        fft_grid = np.zeros(self.vlocal_real.shape, dtype=complex)

        # kinetic
        TPsi_fourier = self.G2 * uG / 2 # (nGvec, )

        # pp_loc + vxc + vHa
        vlocalPsi_fourier = np.zeros((self.nGvec,), dtype=complex) # (nGvec, )
        fft_grid[self.id_cut] = uG
        vlocalPsi_fourier = np.fft.fftn(self.vlocal_real * np.fft.ifftn(fft_grid))[self.id_cut]

        # pp_nonlocal
        vnlocalPsi_fourier = np.zeros((self.nGvec,), dtype=complex) # (nGvec, )
        iatm = 0
        for ityp, natm in enumerate(self.atmlist):
            for _ in range(natm):
                vnl = self.vnl_fourier[iatm]
                vnlocalPsi_fourier += (vnl @ self.Dij[ityp]) @ (vnl.T.conj() @ uG)
                iatm += 1
        vnlocalPsi_fourier /= 2. # Ry to Ht

        HPsi_fourier = vlocalPsi_fourier + vnlocalPsi_fourier + TPsi_fourier # (nGvec,)
        HPsi_fourier[~self.mask] = 0.+0j

        return HPsi_fourier


    def _adjoint(self):
        '''Hamiltonian must be Hermitian!'''
        return self



def scf(cell:SuperCell, 
        e_conv_thr:float=1e-6, 
        maxiter:int=500, 
        mixing:float=0.8, 
        q02:float=1., 
        nbuffer:int=8, 
        xc:str='PBE'):
    # pp_loc real, computed once and cached
    vloc_real = cell.compute_vloc_real()
    
    Dij = []
    for ityp in range(cell.ntyp):
        nidcum = np.concatenate(([0], np.cumsum(cell.pp_nid[ityp]))) # [2,2] => [0,2,4]
        blocks = []
        for l in cell.pp_l[ityp]:
            blocks.append(np.kron(np.eye(2 * l + 1), cell.pp_dij[ityp][nidcum[l]:nidcum[l+1], nidcum[l]:nidcum[l+1]]))
        dij = block_diag(*blocks)
        Dij.append(dij)

    # initial guess
    mask = cell.G2 <= cell.ecutwfc * 2
    uG = gram_schmidt(cell.nGvec, cell.nbnd)
    uG[~mask,:] = 0.+0j
    eden = cell.compute_eden_init_real(do_interpolation=True) # (nx, ny, nz)
    eden = np.clip(eden, a_min=1e-16,a_max=None)
    if xc == 'PBE':
        vxc, _ = cell.compute_vxc_real_pbe(eden)
    elif xc == 'PZ':
        vxc, _ = cell.compute_vxc_real_pz(eden)
    else:
        raise ValueError('PBE or PZ')
    vHa, _ = cell.compute_vHa_real(eden)
    
    # instantiate LinearOperator
    linop = LinOpH(nGvec=cell.nGvec, id_cut=cell.id_cut, G2=cell.G2, Dij=Dij, atmlist=cell.atmlist, ecutwfc=cell.ecutwfc)
    linop.vnl_fourier = cell.compute_vnl_fourier() # once and cached
    
    # preconditioner
    diag_val = 1. / (cell.G2/2 + 1)
    preconditioner = lambda v: diag_val[:,None] * v
    M = LinearOperator(shape=(cell.nGvec, cell.nGvec), dtype=np.dtype(complex), matvec=preconditioner, matmat=preconditioner)

    # Kerker and DIIS
    kerker = np.zeros((cell.nGvec), dtype=float)
    kerker[1:] = cell.G2[1:] / (cell.G2[1:] + q02)
    kerker[0] = 0.
    eden_history = []
    R_history = []
    fft_grid = np.zeros((cell.nx, cell.ny, cell.nz), dtype=complex)


    ##############
    # GAME START #
    ##############
    etot_old = float('inf')
    e_diff   = float('inf')
    counter  = 0
    while abs(e_diff) >= e_conv_thr*2:
        counter += 1

        # update uG and eden
        linop.vlocal_real = vloc_real + vxc + vHa
        w, uG = lobpcg(linop, X=uG, M=M, largest=False, maxiter=200, tol=1e-8) # uG updated
        # w, uG = eigsh(linop, k=cell.nbnd, which='SA', v0=uG[:,0], tol=1e-8, maxiter=100)
        np.testing.assert_allclose(uG.conj().T @ uG, np.identity(cell.nbnd, dtype=complex), atol=1e-12) # check orthonormality
        uG[~mask,:] = 0.+0j

        # Kerker preconditioner + DIIS
        eden_in_G = np.sqrt(cell.V) / cell.N3 * np.fft.fftn(eden)[cell.id_cut] # (nGvec,)
        eden_out_G = np.sqrt(cell.V) / cell.N3 * np.fft.fftn(cell.compute_eden_real(uG))[cell.id_cut] # (nGvec,)
        R_G = kerker * (eden_out_G - eden_in_G)
        if len(eden_history) >= nbuffer:
            eden_history.pop(0)
            R_history.pop(0)
        eden_history.append(eden_in_G)
        R_history.append(R_G)
        if counter <= 1:
            eden_G = eden_in_G + mixing * R_G # Kerker only
            eden_G[0] = cell.nve / np.sqrt(cell.V)
            fft_grid[cell.id_cut] = eden_G
            eden = cell.N3 / np.sqrt(cell.V) * np.fft.ifftn(fft_grid).real
        else:
            n = len(eden_history)
            A = np.zeros((n+1, n+1), dtype=float)
            b = np.zeros((n+1,), dtype=float) ; b[-1] = 1.
            for i in range(n):
                for j in range(n):
                    A[i,j] = np.sum(np.conjugate(R_history[i]) * R_history[j]).real
            A[-1,:] = 1.
            A[:,-1] = -1.
            A[-1,-1] = 0.
            c = np.linalg.solve(A, b)[:n] # drop Lagrange multiplier
            eden_G = np.zeros((cell.nGvec,), dtype=complex)
            for i in range(n):
                eden_G += c[i] * (eden_history[i] + mixing * R_history[i])
            eden_G[0] = cell.nve / np.sqrt(cell.V)
            fft_grid[cell.id_cut] = eden_G
            eden = cell.N3 / np.sqrt(cell.V) * np.fft.ifftn(fft_grid).real

        # eden = (1 - mixing) * cell.compute_eden_real(uG) + mixing * eden # linear mixing
        eden = np.clip(eden, a_min=1e-16,a_max=None)
        nve = np.sum(eden * cell.dV)
        if abs(nve - cell.nve) >= 1e-1:
            warn(f'renormalize nve = {nve}. Target nve = {cell.nve}', )
        eden *= cell.nve / nve # renormalize
        
        # update vxc, vHa
        if xc == 'PBE':
            vxc, exc = cell.compute_vxc_real_pbe(eden)
        elif xc == 'PZ':
            vxc, exc = cell.compute_vxc_real_pz(eden)
        else:
            raise ValueError('PBE or PZ')
        vHa, eHa = cell.compute_vHa_real(eden)
        
        # update T, Vnl
        # <G+k|T ψnk>   = ∑_G unk(G) <G+k|T|G+k> = unk(G) |G+k|^2/2
        # <G+k|Vnl ψnk> = ∑_lm ∑_ij <G+k|βlmi> D^l_ij (∑_G <βlmj|G+k> unk(G))
        TPsi_fourier = cell.G2[:,None] * uG / 2 # (nGvec, nbnd)
        vnlocalPsi_fourier = np.zeros((cell.nGvec, cell.nbnd), dtype=complex) # (nGvec, nbnd)
        iatm = 0
        for ityp, natm in enumerate(cell.atmlist):
            for _ in range(natm):
                vnl = linop.vnl_fourier[iatm]
                vnlocalPsi_fourier += (vnl @ Dij[ityp]) @ (vnl.T.conj() @ uG)
                iatm += 1
        vnlocalPsi_fourier /= 2. # Ry to Ht

        # energy decomposition
        # <O> = ∑_n 2 <ψnk|O ψnk> = ∑_n 2 ∑_G <ψnk|G+k><G+k|O ψnk> = 2 ∑_n ∑_G unk(G)* <G+k|O ψnk>
        eloc = np.sum(vloc_real * eden * cell.dV)
        ekin = 2. * np.sum(uG.conj() * TPsi_fourier).real
        enl = 2. * np.sum(uG.conj() * vnlocalPsi_fourier).real
        e_oe = eloc + ekin + enl
        etot = eloc + ekin + enl + exc + eHa + cell.ewald
        e_diff = etot - etot_old
        print(f'#Iteration {counter:>4d}    total energy = {etot*2:>.14f} Ry    difference = {e_diff*2:>.10f}')
        etot_old = etot

        if counter > maxiter:
            print(f'Not converged after {maxiter} iterations, unconverged total energy = {etot*2} Ry\n\n')
            break
        if abs(e_diff) <= e_conv_thr*2:
            print(f'Total energy        = {etot*2:>.14f} Ry')
            print(f'One particle energy = {e_oe*2:>.8f} Ry')
            print(f'Hartree energy      = {eHa*2:>.14f} Ry')
            print(f'XC energy           = {exc*2:>.14f} Ry')
            print(f'Ewald energy        = {cell.ewald*2:>.14f} Ry\n\n')



def run_all():
    names = ['NH3.cif', 'CH4.cif', 'H2O.cif', 'H2O2.cif', 
             'AlCl3.cif', 'BCl3.cif', 'C2H2.cif', 'C2H4.cif', 
             'CO2.cif', 'CO.cif', 'HCN.cif', 'Li2.cif', 
             'LiH.cif', 'NaCl.cif', 'PH3.cif', 'SO2.cif', 'pyrrole.cif']
    psp_dict = {'C': 'C_ONCV_PBE-1.2.upf',    'H': 'H_ONCV_PBE-1.2.upf',
                'N': 'N_ONCV_PBE-1.2.upf',    'O': 'O_ONCV_PBE-1.2.upf', 
                'S': 'S_ONCV_PBE-1.2.upf',   'Al': 'Al_ONCV_PBE-1.2.upf', 
                'P': 'P_ONCV_PBE-1.2.upf',    'B': 'B_ONCV_PBE-1.2.upf', 
                'Na': 'Na_ONCV_PBE-1.2.upf', 'Cl': 'Cl_ONCV_PBE-1.2.upf', 
                'Be': 'Be_ONCV_PBE-1.2.upf', 'Li': 'Li_ONCV_PBE-1.2.upf', }
    for name in names:
        cell = SuperCell(f'../cif/{name}', ecutwfc=25, ppdir='../pseudo')
        cell.init(psp_dict, verbose=True)
        scf(cell, mixing=0.7, e_conv_thr=1e-6, maxiter=100, q02=0.001)




if __name__ == '__main__':
    run_all()





