# This file is part of the mmd package.
# `multiprocessing` is used to parallelize the calculation of electron repulsion integrals (ERIs).
# DIIS algorithm is implemented to accelerate the SCF convergence.
# Only closeed-shell single point energy calculation at Hartree Fock level implemented, wavefunction can be obtained very straightforwardly, though not implemented yet.
# Author: lyh

import numpy                      as np
from   multiprocessing        import Pool
from   typing                 import Any, TextIO
from   sys                    import stdout
from   molecule               import Molecule, ContractedGaussianFunction
from   utils                  import timeit
from   mmd2                   import overlap, kinetic, potential # implemented in Python; eri also, but rather slow, use mmd.eri or rys.eri_rys instead
from   core.core              import mmd                         # Fortran wrapper, faster than Python implementation
from   core.core              import rys                         # Fortran wrapper for Rys quadrature, can evaluate up to (dd|dd), faster than mmd


def _build_eri(args:list[ContractedGaussianFunction]):
    cgf1, cgf2, cgf3, cgf4 = args
    # return mmd.eri(cgf1.coefs, cgf2.coefs, cgf3.coefs, cgf4.coefs, 
    #                cgf1.expns, cgf2.expns, cgf3.expns, cgf4.expns, 
    #                cgf1.shell, cgf2.shell, cgf3.shell, cgf4.shell, 
    #                cgf1.normalize_factors, cgf2.normalize_factors, cgf3.normalize_factors, cgf4.normalize_factors, 
    #                cgf1.center, cgf2.center, cgf3.center, cgf4.center)

    return rys.eri_rys(cgf1.coefs, cgf2.coefs, cgf3.coefs, cgf4.coefs, 
                       cgf1.expns, cgf2.expns, cgf3.expns, cgf4.expns, 
                       cgf1.shell, cgf2.shell, cgf3.shell, cgf4.shell, 
                       cgf1.normalize_factors, cgf2.normalize_factors, cgf3.normalize_factors, cgf4.normalize_factors, 
                       cgf1.center, cgf2.center, cgf3.center, cgf4.center)
    
    
def build_eri_parallel(cgfs:list[ContractedGaussianFunction], ncgf:int) -> np.ndarray:
    cgf_array = []
    idx = []
    
    for i in range(ncgf):
        for j in range(i+1):
            for k in range(ncgf):
                for l in range(k+1):
                    if (i * (i+1)) // 2 + j >= (k * (k+1)) // 2 + l:
                        cgf_array.append([cgfs[i], cgfs[j], cgfs[k], cgfs[l]])
                        idx.append((i, j, k, l))

    with Pool(processes=8) as pool:
        # parallelize
        result = list(pool.map(_build_eri, cgf_array))
    
    res = np.zeros((ncgf, ncgf, ncgf, ncgf))
    for (i, j, k, l), val in zip(idx, result):
        perms = [(i,j,k,l), 
                 (j,i,k,l), 
                 (i,j,l,k), 
                 (j,i,l,k), 
                 (k,l,i,j), 
                 (l,k,i,j), 
                 (k,l,j,i), 
                 (l,k,j,i)]
        for a, b, c, d in perms:
            res[a, b, c, d] = val    
    return res

    

@timeit
def scf(mol:Molecule, tol=1e-8, maxiter=30, wfn:bool=False, wfn_e:bool=False, log:str|TextIO=stdout) -> Any:
    '''no convergence acceleration technique used'''
    ncgf = 0
    cgfs:list[ContractedGaussianFunction] = []
    for atom in mol.atoms:
        for cgf in atom.cgfs:
            ncgf += 1
            cgfs.append(cgf)
    

    _S = np.zeros((ncgf, ncgf))
    _T = np.zeros((ncgf, ncgf))
    _V = np.zeros((ncgf, ncgf))
    _ERI = np.zeros((ncgf, ncgf, ncgf, ncgf))
    # _ERI = build_eri_parallel(cgfs, ncgf)

    for i in range(ncgf):
        for j in range(ncgf):
            _S[i,j] = overlap(cgfs[i], cgfs[j])
            _T[i,j] = kinetic(cgfs[i], cgfs[j])
    
    for iatom in range(len(mol)):
        for i in range(ncgf):
            for j in range(ncgf):
                _V[i,j] += potential(cgfs[i], cgfs[j], mol.atoms[iatom].xyz) * (-mol.atoms[iatom].number)
        
    '''8-fold eri tensor symmetry'''
    for i in range(ncgf):
        for j in range(i+1):
            for k in range(ncgf):
                for l in range(k+1):
                    print(f"({i:<2},{j:<2},{k:<2},{l:<2})", end='\r')
                    if (i * (i + 1)) // 2 + j >= (k * (k + 1)) // 2 + l:
                        # _tmp = eri(cgfs[i], cgfs[j], cgfs[k], cgfs[l])
                        _tmp = rys.eri_rys(cgfs[i].coefs, cgfs[j].coefs, cgfs[k].coefs, cgfs[l].coefs,
                                           cgfs[i].expns, cgfs[j].expns, cgfs[k].expns, cgfs[l].expns, 
                                           cgfs[i].shell, cgfs[j].shell, cgfs[k].shell, cgfs[l].shell, 
                                           cgfs[i].normalize_factors, cgfs[j].normalize_factors, cgfs[k].normalize_factors, cgfs[l].normalize_factors, 
                                           cgfs[i].center, cgfs[j].center, cgfs[k].center, cgfs[l].center)
                        _ERI[i,j,k,l] = _tmp
                        _ERI[i,j,l,k] = _tmp
                        _ERI[j,i,k,l] = _tmp
                        _ERI[j,i,l,k] = _tmp
                        _ERI[k,l,i,j] = _tmp
                        _ERI[k,l,j,i] = _tmp
                        _ERI[l,k,i,j] = _tmp
                        _ERI[l,k,j,i] = _tmp

    # for i in range(ncgf):
    #     for j in range(ncgf):
    #         for k in range(ncgf):
    #             for l in range(ncgf):
    #                 print(f"({i:<2},{j:<2},{k:<2},{l:<2})", end='\r')
    #                 _ERI[i,j,k,l] = rys.eri_rys(cgfs[i], cgfs[j], cgfs[k], cgfs[l])

    
    Hcore = _T + _V # core hamiltonian

    vals, vecs = np.linalg.eigh(_S)
    S_inv_half = vecs @ np.diag(1. / np.sqrt(vals)) @ vecs.T

    etot_old = float('inf')
    counter = 0
    
    print("Start SCF iteration")

    ne = np.sum(mol.numbers)
    norb = ne // 2
    _F = Hcore.copy()

    while True:
        print(f"Iter {counter:>4}", end='\t')
        Fprime = S_inv_half @ _F @ S_inv_half
        vals, Cprime = np.linalg.eigh(Fprime)
        _C = S_inv_half @ Cprime
        C_occ = _C[:,:norb] # column is the MO coefficient
        _P = 2. * C_occ @ C_occ.T
        # _P = 2. * C_occ @ C_occ.T # density matrix equivalent way # _P = 2. * np.einsum('li,si->ls', C_occ, C_occ, optimize=True) # closed shell restricted, density matrix
        _G = np.einsum('pqrs,rs->pq', _ERI, _P, optimize=True) - 0.5 * np.einsum('prqs,rs->pq', _ERI, _P, optimize=True)
        _F = Hcore + _G # Fock matrix
        etot = 0.5 * np.einsum('pq,pq->', _P, (Hcore + _F), optimize=True)

        print(f"energy difference {etot - etot_old:.10f}" + '\t'*2 + f"energy {etot + mol.e_nuc:.10f}")
        
        if abs(etot - etot_old) <= tol:
            break
        
        counter += 1
        etot_old = etot

        if counter > maxiter:
            print(f'SCF iteration did not coverge in {maxiter} steps!')
            break

    print(f"SCF energy {etot + mol.e_nuc}")



@timeit
def scf_damping(mol:Molecule, tol=1e-8, maxiter=100, wfn:bool=False, wfn_e:bool=False, log:str|TextIO=stdout) -> Any:
    '''simply mix just one old Fock into new Fock with damping factor '''
    ncgf = 0
    cgfs = []
    for atom in mol.atoms:
        for cgf in atom.cgfs:
            ncgf += 1
            cgfs.append(cgf)
    

    _S   = np.zeros((ncgf, ncgf))
    _T   = np.zeros((ncgf, ncgf))
    _V   = np.zeros((ncgf, ncgf))
    _ERI = build_eri_parallel(cgfs, ncgf)

    for i in range(ncgf):
        for j in range(ncgf):
            _S[i,j] = overlap(cgfs[i], cgfs[j])
            _T[i,j] = kinetic(cgfs[i], cgfs[j])
    
    for iatom in range(len(mol)):
        for i in range(ncgf):
            for j in range(ncgf):
                _V[i,j] += potential(cgfs[i], cgfs[j], mol.atoms[iatom].xyz) * (-mol.atoms[iatom].number)
        
    
    Hcore = _T + _V # core hamiltonian

    vals, vecs = np.linalg.eigh(_S)
    S_inv_half = vecs @ np.diag(1. / np.sqrt(vals)) @ vecs.T

    etot_old = float('inf')
    counter  = 0
    
    print("Start SCF iteration")

    ne      = np.sum(mol.numbers)
    norb    = ne // 2
    _F      = Hcore.copy()
    _P_old  = np.zeros((ncgf, ncgf))
    damping = 0.1

    while True:
        print(f"Iter {counter:>4}", end='\t')
        Fprime       = S_inv_half @ _F @ S_inv_half
        vals, Cprime = np.linalg.eigh(Fprime)
        _C           = S_inv_half @ Cprime
        C_occ        = _C[:,:norb]          # column is the MO coefficient
        _P_new       = 2. * C_occ @ C_occ.T # damping
        _P           = damping * _P_old + (1. - damping) * _P_new
        _P_old       = _P_new
        _G           = np.einsum('pqrs,rs->pq', _ERI, _P, optimize=True) - 0.5 * np.einsum('prqs,rs->pq', _ERI, _P, optimize=True)
        _F           = Hcore + _G           # Fock matrix
        etot         = 0.5 * np.einsum('pq,pq->', _P, (Hcore + _F), optimize=True)

        print(f"energy difference {etot - etot_old:.10f}" + '\t'*2 + f"energy {etot + mol.e_nuc:.10f}")
        
        if abs(etot - etot_old) <= tol:
            break
        
        counter += 1
        etot_old = etot

        if counter > maxiter:
            print(f'SCF iteration did not coverge in {maxiter} steps!')
            break

    print(f"SCF energy {etot + mol.e_nuc}")



@timeit
def scf_diis(mol:Molecule, tol=1e-8, maxiter=100, wfn:bool=False, wfn_e:bool=False, log:str|TextIO=stdout) -> Any:
    '''
    Notes
    -----

    Pulay's original DIIS formulation is also known as commutator-DIIS (cDIIS)
        [1] Pulay P (1980) Convergence acceleration of iterative sequences. The case of SCF iteration. Chem Phys Lett 73(2):393-398
        [2] Pulay P (1982) Improved SCF convergence acceleration. J Comput Chem 3(4):556-560
        [3] Lorenzo Maschio (2018) https://doi.org/10.1007/s00214-018-2238-8

    DIIS is usually used after a few common SCF iterations

    `np.linalg.eigh` usually sorts eigenvecs according to eigenvals, if bug comes, check this and sort them if necessary

    if `np.linalg.solve` raises `np.linalg.LinAlgError` due to linear dependence, skip DIIS and change to common iteration
    '''

    ncgf = 0
    cgfs = []
    for atom in mol.atoms:
        for cgf in atom.cgfs:
            ncgf += 1
            cgfs.append(cgf)
    
    _S   = np.zeros((ncgf, ncgf))
    _T   = np.zeros((ncgf, ncgf))
    _V   = np.zeros((ncgf, ncgf))
    _ERI = build_eri_parallel(cgfs, ncgf)

    for i in range(ncgf):
        for j in range(ncgf):
            _S[i,j] = overlap(cgfs[i], cgfs[j])
            _T[i,j] = kinetic(cgfs[i], cgfs[j])
    
    for iatom in range(len(mol)):
        for i in range(ncgf):
            for j in range(ncgf):
                _V[i,j] += potential(cgfs[i], cgfs[j], mol.atoms[iatom].xyz) * (-mol.atoms[iatom].number)

    Hcore = _T + _V # core hamiltonian

    vals, vecs = np.linalg.eigh(_S)
    S_inv_half = vecs @ np.diag(1. / np.sqrt(vals)) @ vecs.T # hermitian

    etot_old = float('inf')
    counter  = 0
    
    print("Start SCF iteration")

    ne       = np.sum(mol.numbers)
    norb     = ne // 2
    _F       = Hcore.copy()
    Fs       = []
    diis_res = []
    nbuffer  = 6 # Fock list and residue list should not exceed diis_vec_maxlen
    
    while True:
        print(f"Iter {counter:>4}", end='\t')
        Fprime       = S_inv_half @ _F @ S_inv_half
        vals, Cprime = np.linalg.eigh(Fprime)
        _C           = S_inv_half @ Cprime
        C_occ        = _C[:,:norb]          # column is the MO coefficient
        _P           = 2. * C_occ @ C_occ.T # density matrix, other notation maybe D
        _G           = np.einsum('pqrs,rs->pq', _ERI, _P, optimize=True) - 0.5 * np.einsum('prqs,rs->pq', _ERI, _P, optimize=True)
        _F           = Hcore + _G           # Fock matrix

        etot         = 0.5 * np.einsum('pq,pq->', _P, (Hcore + _F), optimize=True)

        print(f"energy difference {etot - etot_old:.10f}" + '\t'*2 + f"energy {etot + mol.e_nuc:.10f}")
        
        if abs(etot - etot_old) <= tol:
            break
        
        counter += 1
        etot_old = etot

        if counter > maxiter:
            print(f'SCF iteration did not coverge in {maxiter} steps!')
            break

        if counter <= 2:
            # DIIS usually starts after a few iterations
            continue
        
        # ---------------- #
        # DIIS starts here #
        # ---------------- #
        if len(Fs) > nbuffer:
            Fs.pop(0)
            diis_res.pop(0)
        diis_res.append(S_inv_half @ ((_F @ _P @ _S) - _S @ _P @ _F) @ S_inv_half)
        Fs.append(_F)

        _B         = np.zeros((len(Fs)+1, len(Fs)+1))
        _B[-1, :]  = -1.
        _B[:, -1]  = -1.
        _B[-1, -1] =  0.

        for i in range(len(Fs)):
            for j in range(len(Fs)):
                 _B[i,j] += np.trace(diis_res[i].T @ diis_res[j])

        diis_rhs     = np.zeros((len(_B), ))
        diis_rhs[-1] = -1.
        try:
            # if Pulay equation is nearly linear dependent, linear solve may not work
            diis_coef = np.linalg.solve(_B, diis_rhs)
        except np.linalg.LinAlgError:
            print(f"DIIS solve Pulay equation failed, last step Fock matrix is resoted")
            continue

        _F = np.zeros(_F.shape)
        for idx in range(len(diis_coef)-1):
            _F += Fs[idx] * diis_coef[idx]


    print(f"SCF energy {etot + mol.e_nuc}")




if __name__ == '__main__':
    # mol = Molecule('./data/Ethene.xyz', './data/cc-pvdz.dat') # when use aug-cc-pvdz, scf without damping does not converge
    # scf_diis(mol)

    pass