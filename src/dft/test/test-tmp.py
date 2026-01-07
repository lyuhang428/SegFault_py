import sys
sys.path.insert(0, '../src/')
import numpy as np
import pylibxc
from pyqint import MoleculeBuilder, cgf
from pydft import MolecularGrid, DFT

slaterX = pylibxc.LibXCFunctional('lda_x', 'unpolarized')
vwn5 = pylibxc.LibXCFunctional('lda_c_vwn_rpa', 'unpolarized')
    
def main():
    mol = MoleculeBuilder().from_file('../data/xyz/co2.xyz')
    cgfs, atoms = mol.build_basis('sto3g')
    molgrid = MolecularGrid(atoms, cgfs, nshells=32, nangpts=110)
    
    molgrid.initialize()
    gridpoints = molgrid.get_grid_coordinates()

    dft = DFT(mol, basis='sto3g')
    en = dft.scf(1e-6, ) ; print(f'en: {en}')
    molgrid_copy = dft.get_molgrid_copy() # `molgrid_copy` and `molgrid` is the same
    grid = molgrid.get_grid_coordinates()
    aos_vals = []
    for i in range(len(cgfs)):
        aos_vals.append(molgrid.get_amplitude_at_points(np.concatenate(grid), np.pad([1.], (i,len(cgfs)-i-1), 'constant', constant_values=(0.,0.))))
    aos_vals = np.array(aos_vals) ; np.savetxt('/home/lyh/tmp/.aos_vals.dat', aos_vals, fmt="%.20f")
    S = dft.get_data()['S'] ; np.savetxt('/home/lyh/tmp/.S.dat', S, fmt="%.20f")
    T = dft.get_data()['T'] ; np.savetxt('/home/lyh/tmp/.T.dat', T, fmt="%.20f")
    V = dft.get_data()['V'] ; np.savetxt('/home/lyh/tmp/.V.dat', V, fmt="%.20f")
    J = dft.get_data()['J'] ; np.savetxt('/home/lyh/tmp/.J.dat', J, fmt="%.20f")
    XC = dft.get_data()['XC'] ; np.savetxt('/home/lyh/tmp/.XC.dat', XC, fmt="%.20f")
    P = dft.get_data()['P'] ; np.savetxt('/home/lyh/tmp/.P.dat', P, fmt="%.20f")
    F = dft.get_data()['F'] ; np.savetxt('/home/lyh/tmp/.F.dat', F, fmt="%.20f")
    C = dft.get_data()['C'] ; np.savetxt('/home/lyh/tmp/.C.dat', C, fmt="%.20f")
    
    np.testing.assert_allclose(F, (T+V+J+XC))
    print(np.trace(P @ J)/2)
    print(dft.get_data()['Exc'])
    print(abs(2 * C[:,:7] @ C[:,:7].T - P).max())

    


if __name__ == '__main__':
    main()