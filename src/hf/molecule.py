# This file is part of homemade-hf, a simple Hartree-Fock implementation.
# This file contains the `Molecule`, `Atom`, and `ContractedGaussianFunction` classes
# When building a `molecule`, it reads the xyz file and basis set file, and creates `atoms` with their `contracted Gaussian functions`.
# The overall structure is: Molecule -> Atom -> ContractedGaussianFunction
# Author: lyh

import re
import numpy as np
import ase
from ase.io import read
from utils import safe_factorial2


class BasisSetNotAvailableError(Exception):
    def __init__(self, msg:str='no basis set found for this element', *args):
        super().__init__(*args)
        self.msg = msg
    
    def __str__(self):
        return self.msg


class Molecule:
    '''
    Attributes
    ----------
    self.numbers : list
        nuclear charge of all atoms

    self.xyz : list | np.ndarray
        coordinates of all atoms

    self.symbols : list[str]
        element symbols of all atoms

    self.natom : int
        number of atoms

    self.atoms : list[Atom]
        Atom objects

    Parameters
    ----------
    xyz_filename : str
        'h2o.xyz' for example

    bas_filaneme : str
        'cc-pvdz.dat' for example. Assignment of different basis to different atoms is supported
    '''

    def __init__(self, xyz_filename:str, bas_filename:str):
        mol          : ase.Atoms  = read(xyz_filename, index=0)
        self.symbols : list       = mol.get_chemical_symbols()
        self.numbers : list       = mol.get_atomic_numbers() # nuclear charges of each atoms
        self.xyz     : np.ndarray = mol.positions            # in angstrom
        self.xyz_    : np.ndarray = self._centerlize()       # in angstrom
        self.xyz__   : np.ndarray = self.ang2bohr()          # in bohr, centralized
        self.natom   : int        = len(mol)
        self.atoms   : list[Atom] = []
        self.e_nuc   : float      = self.nuclear_repulsion() # classical nuclear repulsion energy [Eh], e = \sum_{A\ne B} (Z_A Z_B) / R_{AB}

        for symbol, number, xyz in zip(self.symbols, self.numbers, self.xyz__):
            self.atoms.append(Atom(bas=bas_filename, symbol=symbol, number=number, xyz=xyz))
    
    def __len__(self) -> int:
        return self.natom
    
    def _centerlize(self) -> np.ndarray:
        shift = self.xyz.mean(axis=0)
        return self.xyz - shift
    
    def ang2bohr(self) -> np.ndarray:
        return self.xyz_ / 0.52917720859
    
    def nuclear_repulsion(self) -> float:
        e_nuc = 0.
        for i in range(self.natom-1):
            for j in range(i+1, self.natom):
                e_nuc += self.numbers[i] * self.numbers[j] / np.linalg.norm(self.xyz__[i] - self.xyz__[j])
        return e_nuc



class Atom:
    '''
    one single atom and all cgfs it has

    Attributes
    ----------

    self.bas: str
        basis set filename, default 'cc-pvdz.dat'

    self.symbol: str
        element symbol

    self.number : int
        nuclear charge, or atomic number

    self.cgfs : list[ContractedGaussianFunction]
        cgf list

    self.xyz : list
        xyz coordinate of this atom
    '''
    _orbitals = ['S', 'P', 'D', 'F']

    def __init__(self, bas:str, symbol:str, number:int, xyz:list|np.ndarray):
        self.bas    : str                              = bas
        self.symbol : str                              = symbol
        self.number : int                              = number
        self.xyz    : list | np.ndarray                = xyz
        self.cgfs   : list[ContractedGaussianFunction] = []

        bas = open(self.bas, 'r').read().split(sep='****')
        target = ''
        for bb in bas:
            block = bb.strip()
            if re.match(rf'^{self.symbol}\s+0', block) is not None:
                target += block
                break
        
        if not target:
            raise BasisSetNotAvailableError()
        
        target = target.split(sep='\n') # list
        
        for idx, line in enumerate(target):
            if line[0] in Atom._orbitals:
                orbital_type = line[0]
                expns = []
                coefs = []
                n = int(line.split()[1])
                for token in target[idx+1: idx+1+n]:
                    expn, coef = list(map(float, token.strip().split()))
                    expns.append(expn)
                    coefs.append(coef)
                if orbital_type == 'S':
                    # s
                    self.cgfs.append(ContractedGaussianFunction(npgf=n, expns=expns, coefs=coefs, orbital_type=orbital_type, shell=[0,0,0], center=self.xyz))
                elif orbital_type == 'P':
                    # px -> [1,0,0]; py -> [0,1,0]; pz -> [0,0,1]
                    self.cgfs.append(ContractedGaussianFunction(npgf=n, expns=expns, coefs=coefs, orbital_type=orbital_type, shell=[1,0,0], center=self.xyz))
                    self.cgfs.append(ContractedGaussianFunction(npgf=n, expns=expns, coefs=coefs, orbital_type=orbital_type, shell=[0,1,0], center=self.xyz))
                    self.cgfs.append(ContractedGaussianFunction(npgf=n, expns=expns, coefs=coefs, orbital_type=orbital_type, shell=[0,0,1], center=self.xyz))
                elif orbital_type == 'D':
                    # dxx, dxy, dxz, dyy, dyz, dzz
                    self.cgfs.append(ContractedGaussianFunction(npgf=n, expns=expns, coefs=coefs, orbital_type=orbital_type, shell=[2,0,0], center=self.xyz))
                    self.cgfs.append(ContractedGaussianFunction(npgf=n, expns=expns, coefs=coefs, orbital_type=orbital_type, shell=[1,1,0], center=self.xyz))
                    self.cgfs.append(ContractedGaussianFunction(npgf=n, expns=expns, coefs=coefs, orbital_type=orbital_type, shell=[1,0,1], center=self.xyz))
                    self.cgfs.append(ContractedGaussianFunction(npgf=n, expns=expns, coefs=coefs, orbital_type=orbital_type, shell=[0,2,0], center=self.xyz))
                    self.cgfs.append(ContractedGaussianFunction(npgf=n, expns=expns, coefs=coefs, orbital_type=orbital_type, shell=[0,1,1], center=self.xyz))
                    self.cgfs.append(ContractedGaussianFunction(npgf=n, expns=expns, coefs=coefs, orbital_type=orbital_type, shell=[0,0,2], center=self.xyz))
                elif orbital_type == 'F':
                    # fxxx, fxxy, fxxz, fxyy, fxyz, fxzz, fyyy, fyyz, fyzz, fzzz
                    self.cgfs.append(ContractedGaussianFunction(npgf=n, expns=expns, coefs=coefs, orbital_type=orbital_type, shell=[3,0,0], center=self.xyz))
                    self.cgfs.append(ContractedGaussianFunction(npgf=n, expns=expns, coefs=coefs, orbital_type=orbital_type, shell=[2,1,0], center=self.xyz))
                    self.cgfs.append(ContractedGaussianFunction(npgf=n, expns=expns, coefs=coefs, orbital_type=orbital_type, shell=[2,0,1], center=self.xyz))
                    self.cgfs.append(ContractedGaussianFunction(npgf=n, expns=expns, coefs=coefs, orbital_type=orbital_type, shell=[1,2,0], center=self.xyz))
                    self.cgfs.append(ContractedGaussianFunction(npgf=n, expns=expns, coefs=coefs, orbital_type=orbital_type, shell=[1,1,1], center=self.xyz))
                    self.cgfs.append(ContractedGaussianFunction(npgf=n, expns=expns, coefs=coefs, orbital_type=orbital_type, shell=[1,0,2], center=self.xyz))
                    self.cgfs.append(ContractedGaussianFunction(npgf=n, expns=expns, coefs=coefs, orbital_type=orbital_type, shell=[0,3,0], center=self.xyz))
                    self.cgfs.append(ContractedGaussianFunction(npgf=n, expns=expns, coefs=coefs, orbital_type=orbital_type, shell=[0,2,1], center=self.xyz))
                    self.cgfs.append(ContractedGaussianFunction(npgf=n, expns=expns, coefs=coefs, orbital_type=orbital_type, shell=[0,1,2], center=self.xyz))
                    self.cgfs.append(ContractedGaussianFunction(npgf=n, expns=expns, coefs=coefs, orbital_type=orbital_type, shell=[0,0,3], center=self.xyz))

    def __len__(self):
        return len(self.cgfs)




class ContractedGaussianFunction:
    '''
    one single contracted gaussian function

    Parameters
    ----------

    self.npgf : int
        num of primitive gaussian in one contracted gaussian function

    self.expn : list[float]
        exponent of each pgf
    
    self.coef : list[float]
        coefficient of each pgf
    
    self.orbital_type : str
        S, P, D, F, G, H, I.... Different types have different shells

    self.shell : list[int]
        Angular part of AO: x^lx y^ly z^lz. s -> [0,0,0], px -> [1,0,0], py -> [0,1,0], ..., dxx -> [2,0,0], ...
    
    self.center : list[float] | np.ndarray[float]
        xyz coordinate this cgf centers at

    self.normalize_factors : list
        normalization factor for each pgf in a cgf
    '''
    def __init__(self, npgf:int, expns:list, coefs:list, orbital_type:str, shell:list, center:list):
        self.npgf              = npgf
        self.expns             = expns
        self.coefs             = coefs
        self.orbital_type      = orbital_type
        self.shell             = shell
        self.center            = center
        self.normalize_factors = []

        l = self.shell[0] + self.shell[1] + self.shell[2]
        self.normalize_factors = np.sqrt(np.power(2, 2 * l + 1.5) * 
                                         np.power(self.expns, l + 1.5) / 
                                         safe_factorial2(2 * self.shell[0] - 1) / 
                                         safe_factorial2(2 * self.shell[1] - 1) / 
                                         safe_factorial2(2 * self.shell[2] - 1) / 
                                         np.power(np.pi, 1.5))
        prefactor = np.power(np.pi, 1.5) * safe_factorial2(2 * self.shell[0] - 1) * safe_factorial2(2 * self.shell[1] - 1) * safe_factorial2(2 * self.shell[2] - 1) / np.power(2., l)
        N = 0.
        num_exps = len(self.expns)
        for ia in range(num_exps):
            for ib in range(num_exps):
                N += self.normalize_factors[ia] * self.normalize_factors[ib] * self.coefs[ia] * self.coefs[ib] / np.power(self.expns[ia] + self.expns[ib], l + 1.5)
        N *= prefactor
        N = np.power(N, -0.5)
        for ia in range(num_exps):
            self.coefs[ia] *= N



class WFN:
    ...
    # TODO: buildup S, T, V, ERI like `wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))`
    # TODO: mints = psi4.core.MintsHelper(wfn.basisset())
    # TODO: S = mints.ao_overlap()



if __name__ == "__main__":
    pass






