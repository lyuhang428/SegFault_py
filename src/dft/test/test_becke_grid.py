import sys
sys.path.insert(0, '../src/')
import numpy as np
import pylibxc
from lbeckeGrid import BeckeFuzzyCell, gaussCheby2
from lrsh import RSH_CART, lm
from loeoperator import Molecule, Overlap, Kinetic, External

slaterX = pylibxc.LibXCFunctional('lda_x', 'unpolarized')
vwn5 = pylibxc.LibXCFunctional('lda_c_vwn_rpa', 'unpolarized')

def test_becke_cell():
    N = 150
    sz = 8
    x = np.linspace(-sz,sz,N)
    xv,yv = np.meshgrid(x,x)
    points = np.array([[x,y,0] for x,y in zip(xv.flatten(),yv.flatten())])

    # me
    mymol = Molecule('../data/xyz/benzene.xyz')
    becke = BeckeFuzzyCell(mymol.symbols, mymol.rms, mymol.numbers, mymol.xyz, ncheb=32, nleb=17, k=3, biased=True)
    myweight = becke.get_weight_p(points)
    np.testing.assert_allclose(myweight.sum(axis=0), 1.)

    # ifilot
    from pydft import MolecularGrid
    from pyqint import MoleculeBuilder
    mol = MoleculeBuilder().from_file('../data/xyz/benzene.xyz')
    cgfs, atoms = mol.build_basis('sto3g')
    molgrid = MolecularGrid(atoms, cgfs)
    mweights = molgrid.calculate_weights_at_points(points, k=3)

    # ---------------------------------------
    for i in range(mymol.natom):
        if mymol.symbols[i] != mol.get_atoms()[i][0]:
            raise Exception
        np.testing.assert_allclose(mymol.xyz[i], mol.get_atoms()[i][1])
    
    # ---------------------------------------
    import matplotlib.pyplot as plt
    plt.figure(dpi=144)
    plt.imshow(np.max(myweight,axis=0).reshape((N,N)),
           extent=(-sz,sz,-sz,sz), interpolation='bicubic')
    plt.xlabel('x [a.u.]')
    plt.ylabel('y [a.u.]')
    plt.colorbar()
    plt.grid(linestyle='--', color='black', alpha=0.5)

    # add the atoms to the plot
    r = np.zeros((len(atoms), 3))
    for i,at in enumerate(atoms):
        r[i] = at[0]
    plt.scatter(r[0:6,0], r[0:6,1], s=50.0, color='grey', edgecolor='black')
    plt.scatter(r[6:12,0], r[6:12,1], s=50.0, color='white', edgecolor='black')

    plt.tight_layout()
    plt.show()

    

def test_becke_cell2():
    points = np.random.random((10000,3)) * 100.

    # me
    mymol = Molecule('../data/xyz/co.xyz', basefile='sto-3g')
    aos = mymol.aos
    olp = Overlap(aos)
    kin = Kinetic(aos)
    ext = External(aos, mymol.xyz, mymol.numbers)
    sij = olp.Sij()
    tij = kin.Tij()
    vij = ext.Vij()
    hij = tij + vij
    np.testing.assert_allclose(sij, sij.T)
    np.testing.assert_allclose(tij, tij.T)
    np.testing.assert_allclose(vij, vij.T)
    np.testing.assert_allclose(hij, hij.T)
    becke = BeckeFuzzyCell(mymol.symbols, mymol.rms, mymol.numbers, mymol.xyz, ncheb=32, nleb=17, k=3, biased=True)
    grid, grid_global = becke.build_grid()
    natom, nrad, nang, _ = grid.shape
    nao = len(aos)
    ne  = mymol.numbers.sum()
    nlm = len(lm)
    
    y_jk = np.zeros((nang, nlm)) # 球諧基
    for ilm, (l, m) in enumerate(lm):
        y_jk[:,ilm] = RSH_CART[f'{l}'][f'{m}'](*becke.xleb)
    for ilm, (l,m) in enumerate(lm): # 正交歸一性檢查
        for iilm, (ll,mm) in enumerate(lm):
            integral = np.sum(y_jk[:,ilm] * y_jk[:,iilm] * becke.wleb)
            if (l == ll) and (m == mm):
                if not np.isclose(integral, 1.):
                    raise Exception
            else:
                if not np.isclose(integral, 0.):
                    raise Exception

    myweights = becke.get_weight_p(points)
    mylmprojection = np.zeros((natom, nrad, nlm))


    # ifilot
    from pydft import MolecularGrid, DFT
    from pyqint import MoleculeBuilder
    mol = MoleculeBuilder().from_name('co')
    cgfs, atoms = mol.build_basis('sto3g')
    molgrid = MolecularGrid(atoms, cgfs, nshells=32, nangpts=110)
    mweights = molgrid.calculate_weights_at_points(points)
    np.testing.assert_allclose(myweights , mweights, atol=1e-15)
    
    molgrid.initialize()
    gridpoints = molgrid.get_grid_coordinates()

    for i in range(natom):
        mygrid = []
        for j in range(nrad):
            mygrid.extend(grid[i,j,:,:3])
        np.testing.assert_allclose(grid[i,:,:,3+i].ravel(), molgrid.get_becke_weights()[i], atol=1e-15)
        np.testing.assert_allclose(                 mygrid,                  gridpoints[i], atol=1e-15)

    dft = DFT(mol, basis='sto3g')
    en = dft.scf(1e-6, ) ; print(f'en: {en}')
    molgrid_copy = dft.get_molgrid_copy() # `molgrid_copy` and `molgrid` is the same
    for i in range(natom):
        mygrid = []
        for j in range(nrad):
            mygrid.extend(grid[i,j,:,:3])
        np.testing.assert_allclose(   molgrid_copy.get_becke_weights()[i], grid[i,:,:,i+3].ravel(), atol=1e-15)
        np.testing.assert_allclose(molgrid_copy.get_grid_coordinates()[i],                 mygrid , atol=1e-15)
    lmprojection = dft.get_molgrid_copy().get_rho_lm_atoms()
    edens = dft.get_molgrid_copy().get_densities()
    print(lmprojection.shape)
    print(len(edens), edens[0].shape)

    _ylm = np.loadtxt('/home/lyh/tmp/._ylm.dat', dtype=float)
    print(abs(_ylm - y_jk.T).max())
    
    ne = 0.
    for i in range(natom): # 從收斂的電子密度重構出電子數
        ne += np.dot((becke.rcheb[i]**2 * becke.wcheb[i]), (edens[i].reshape((nrad, nang)) * grid[i,:,:,i+3]) @ becke.wleb)
        mylmprojection[i] = (edens[i].reshape((nrad, nang)) * grid[i,:,:,i+3]) @ (y_jk * becke.wleb[:,None]) * becke.rcheb[i][:,None]**2 # 爲什麼要乘r^2?
    print(f'{ne:.8f}')
    np.testing.assert_allclose(lmprojection, mylmprojection, atol=1e-14)

    print(abs(sij - dft.get_data()['S']).max(), abs(sij - dft.get_data()['S']).min())
    print(abs(tij - dft.get_data()['T']).max(), abs(tij - dft.get_data()['T']).min())
    print(abs(vij - dft.get_data()['V']).max(), abs(vij - dft.get_data()['V']).min())
    print(mymol.e_nuc - dft.get_data()['enucrep'])
    
    x = np.linspace(-2, 2, 100) # produce meshgrid for the xz-plane
    iz, ix = np.meshgrid(x, x, indexing='ij')
    points = np.stack((ix.ravel(), [0.] * 10000, iz.ravel()), axis=1) # y=0
    
    myaos_vals = np.zeros((nao, len(points)))
    for i in range(nao):
        myaos_vals[i] = aos[i].ao_val_p(*points.T)
    myedens = np.diag(myaos_vals.T @ dft.get_data()['P'] @ myaos_vals) ; myedens = myedens.ravel()
    
    fieldx = molgrid_copy.get_exchange_potential_at_points(points, dft.get_data()['P']) # 交換勢
    fieldc = molgrid_copy.get_correlation_potential_at_points(points, dft.get_data()['P']) # 交換勢
    # field = field.reshape((100, 100))
    print(abs(myaos_vals - np.loadtxt("/home/lyh/tmp/.amps.dat", dtype=float)).max())
    print(abs(myedens - np.loadtxt('/home/lyh/tmp/.dens.dat', dtype=float)).max())

    inp = {'rho': myedens}
    myvx = slaterX.compute(inp)['vrho'].ravel()
    myvc = vwn5.compute(inp)['vrho'].ravel()
    np.testing.assert_allclose(myvx, fieldx, atol=1e-12)
    np.testing.assert_allclose(myvc, fieldc, atol=1e-12)
    
    myaos_vals = np.zeros((nao, natom*nrad*nang))
    for i in range(nao):
        for j in range(natom):
            myaos_vals[i, j*nrad*nang:(j+1)*nrad*nang] = aos[i].ao_val_p(grid[j,:,:,0].ravel(), grid[j,:,:,1].ravel(), grid[j,:,:,2].ravel())
    print(abs(myaos_vals - np.loadtxt('/home/lyh/tmp/.fullgrids_amp.dat')).max())

    # ----------------------------------------------



if __name__ == '__main__':
    # test_becke_cell()
    test_becke_cell2()