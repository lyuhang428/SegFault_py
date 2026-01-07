# -*- encoding: utf-8 -*-
# lyh
# functions prefixed with '__' are helper functions, they should not be used outside this file
import time
import numpy as np
import pylibxc
from   scipy.interpolate import CubicSpline
from   lrsh              import rsh
from   lbeckeGrid        import BeckeFuzzyCell, poisson_solver, timeit
from   loeoperator       import Molecule, Overlap, Kinetic, External, AO
from   sys               import exit

slaterX = pylibxc.LibXCFunctional('lda_x',     'unpolarized')
vwn5    = pylibxc.LibXCFunctional('lda_c_vwn_rpa', 'unpolarized')

def _canonical_orthogolization(F:np.ndarray, S:np.ndarray) -> np.ndarray:
    '''Solve FC = SCe, return C. F and S should be real symmetric'''
    s, u = np.linalg.eigh(S)
    x = u @ np.linalg.diag(np.sqrt(1. / s))
    fprime = x.T @ F @ x
    e, cprime = np.linalg.eigh(fprime)
    C = x @ cprime
    if abs(F @ C - S @ C @ np.diag(e)).max() > 1e-12:
        raise ValueError("Failed")
    return C

def _symmetric_orthogonalization(F:np.ndarray, S:np.ndarray) -> np.ndarray:
    '''Solve FC = SCe, return C. F and S should be real symmetric'''
    s, u = np.linalg.eigh(S)
    s_inv_half = u @ np.diag(np.sqrt(1. / s)) @ u.T
    fprime = s_inv_half @ F @ s_inv_half
    e, cprime = np.linalg.eigh(fprime)
    C = s_inv_half @ cprime
    if abs(F @ C - S @ C @ np.diag(e)).max > 1e-12:
        raise ValueError('Failed')
    return C

# @timeit
def _get_ao_vals_on_grid(aos:list[AO], grid_global:np.ndarray) -> np.ndarray:
    '''
    計算所有基函數在所有網格點上的函數值. `BeckeFuzzyCell.build_grid -> grid_global`. 
    
    計算在任意座標處的基函數值請使用`AO.ao_val_p/s`.

    Arguments
    ---------
    aos : list[AO]
        `AO` instances list.
    grid_global : np.ndarray (natom, nradxnang, 3)
        Different from `BeckeFuzzyCell.build_grid->grid` (natom,nrad,nang,3+natom), `grid_global` 
        is flattened and stores only coordinates.
    
    Return
    ------
    aos_vals : np.ndarray (nao, natom, nradxnang)
    '''
    natom, natgrid, _ = grid_global.shape # (natom, nradxnang, 3)
    nao = len(aos)
    aos_vals = np.zeros((nao, natom, natgrid))
    for iao in range(nao):
        for iatom in range(natom):
            aos_vals[iao,iatom] = aos[iao].ao_val_p(x=grid_global[iatom,:,0], y=grid_global[iatom,:,1], z=grid_global[iatom,:,2])
    return aos_vals

# @timeit
def _get_distance_and_solid_angle(grid_global:np.ndarray, xyz:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''計算所有網格點到各原子的距離與固體角. 原子在自身產生的網格點上的距離和固體角也包含在內. 用於插值和重構整個實空間上的庫倫勢. 

    Arguments
    ---------
    grid_global : np.ndarray (natom,nradxnang,3)
        Must be generated from `BeckeFuzzyCell.build_grid`.
    xyz : np.ndarray (natom,3)
        Molecular coordinates, must be given from `Molecule.xyz`.

    Return
    ------
    dist : np.ndarray (natom,natomxnradxnang)
        Each row is the distance of all grid points to the row-th atom.
    theta : np.ndarray (natom,natomxnradxnang)
        Each row is the theta angle of all grid points to the row-th atom.
    phi : np.ndarray (natom,natomxnradxnang)
        Same as theta.
    '''
    grid_total = np.concatenate(grid_global) # (natomxnradxnang,3)
    natom = len(xyz)
    ngrid = len(grid_total)
    dist  = np.zeros((natom,ngrid))
    theta = np.zeros((natom,ngrid))
    phi   = np.zeros((natom,ngrid))
    for iatom in range(natom):
        _disp = grid_total[:,:3] - xyz[iatom]
        _dist = np.linalg.norm(_disp, axis=1)
        _disp_normalized = _disp / _dist[:,None]
        _theta = np.arccos(_disp_normalized[:,2])
        _phi = np.arctan2(_disp_normalized[:,1], _disp_normalized[:,0])
        dist[iatom]  = _dist
        theta[iatom] = _theta
        phi[iatom]   = _phi

    return dist, theta, phi

# @timeit
def _get_spherical_vals(lm:list[(int,int)], theta:np.ndarray, phi:np.ndarray) -> np.ndarray:
    '''
    Pre-compute Ylm(solid_angle) of all grid points relative to each atom (all theta and phi) up to lmax.
    `theta` and `phi` must be generated from `_build_distance_and_solid_angle`

    Arguments
    ---------
    lm : list[(int,int)]
        Angular number.
    theta : np.ndarray (natom, natomxnradxnang)
        theta angle of all grid points relative to each atom.
    phi : np.ndarray (natom, natomxnradxnang)
        same as theta.

    Return
    ------
    ylm : np.ndarray (natom, nlm, natomxnradxnang)
    '''
    if theta.shape != phi.shape:
        raise Exception
    natom, ngrid = theta.shape
    nlm = len(lm)
    ylm = np.zeros((natom,nlm,ngrid)) # (natom,natomxnradxnang)

    for iatom in range(natom):
        for ilm, (l,m) in enumerate(lm):
            ylm[iatom,ilm] = rsh(l, m, theta=theta[iatom], phi=phi[iatom])
    
    return ylm

# @timeit
def _build_spherical_basis(lm:list[(int,int)], x:np.ndarray, y:np.ndarray, z:np.ndarray) -> np.ndarray:
    r'''
    構建單位球面上的球諧基. 所有原子共用一套. 

    Arguments
    ---------
    lm : list[(int,int)]
        Angular number. (lmin, -lmin\~lmin), ..., (lmax, -lmax\~lmax)
    x, y, z : np.ndarray:
        Cartesian coordinates on unit sphere. Must be generated from `BeckeFuzzyCell.xleb`

    Return
    ------
    y_jk : np.ndarray (nang,nlm)
    '''
    nlm = len(lm)
    nang = len(x)
    y_jk = np.zeros((nang,nlm))
    for ilm, (l,m) in enumerate(lm):
        y_jk[:,ilm] = rsh(l, m, theta=np.arccos(z), phi=np.arctan2(y,x))
    return y_jk

@timeit
def dft(xyzfile:str, basefile:str, 
        maxiter:int=30, e_convergence:float=1e-6, d_convergence:float=1e-6, 
        nbuffer:int=-1, initial_guess:str='sap', radial_scheme:str='becke', 
        radial_points:int=75, angular_order:int=29, k:int=4, biased:bool=True):
    beg = time.time()
    
    mol = Molecule(xyzfile=xyzfile, basefile=basefile)
    aos = mol.aos
    ne  = mol.numbers.sum()
    nao = len(aos)

    olp = Overlap(aos)
    kin = Kinetic(aos)
    ext = External(aos, mol.xyz, mol.numbers)
    sij = olp.Sij() ; np.testing.assert_allclose(sij, sij.T)
    tij = kin.Tij() ; np.testing.assert_allclose(tij, tij.T)
    vij = ext.Vij() ; np.testing.assert_allclose(vij, vij.T)
    hij = tij + vij ; np.testing.assert_allclose(hij, hij.T)

    # 構建分子網格
    becke = BeckeFuzzyCell(symbols=mol.symbols, rms=mol.rms, numbers=mol.numbers, xyz=mol.xyz, ncheb=radial_points, nleb=angular_order, k=k, biased=biased)
    lmax = angular_order//2 # 14
    lm = [(l,m) for l in range(lmax+1) for m in range(-l,l+1)]
    nlm = len(lm)
    grid, grid_global = becke.build_grid() # (natom, nrad, nang, 3+natom), (natom, nradxnang, 3)
    natom, nrad, nang, _ = grid.shape
    mweights = np.array([np.outer(becke.rcheb[iatom]**2 * becke.wcheb[iatom], becke.wleb).ravel() * becke.weight[iatom] for iatom in range(natom)])
    
    dist, theta, phi = _get_distance_and_solid_angle(grid_global, mol.xyz) # (natom, natomxnradxnang)
    zz = [(becke.ncheb + 1) / np.pi * np.arccos((dist[iatom] - mol.rms[iatom]) / (dist[iatom] + mol.rms[iatom])) for iatom in range(natom)] # 用於插值
    mask_ubound = [zz[iatom] > becke.ncheb for iatom in range(natom)] # 用於外推
    mask_lbound = [zz[iatom] < 1.          for iatom in range(natom)] # 用於外推
    aos_vals = _get_ao_vals_on_grid(aos, grid_global) # (nao, natom, nradxnang) 计算原子轨道在网格上的值
    y_jk = _build_spherical_basis(lm, *becke.xleb) # (nang, nlm) 單位球面球谐基
    ylm = _get_spherical_vals(lm, theta, phi) # 所有網格點相對於各原子球諧函數值. 很慢, 不知道爲什麼

    print(f"Number of atoms          {natom:<8d}")
    print(f"Number of electrons      {ne:<8d}")
    print(f"Number of aos            {nao:<8d}")
    print(f"Number of radial grid    {nrad:<8d}")
    print(f"Number of angular grid   {nang:<8d}")
    print(f"Number of total grid     {natom*nrad*nang:<8d}")
    print(f"Grid dimension           {grid.shape}")
    print(f"Maximum angular momentum lmax={lmax}")
    print(f"Nuclear repulsion energy {mol.e_nuc:<15.10f} a.u.")
    print()

    print(f"{time.time() - beg:.8f} sec")

    # etot_old = float('inf')
    etot_old = float('nan')
    Puv_old = float('nan')
    focks = []
    diis_res = []
    counter = 0

    # initial guess
    # Susi Lehtola, J. Chem. Theory Comput. 2019, 15, 1593−1604, Superposition of Atomic Potential
    s, u = np.linalg.eigh(sij)
    sij_inv_half = u @ np.diag(np.sqrt(1. / s)) @ u.T # S^{-1/2} Symmetric orthogonalization
    if initial_guess.lower() == 'core':
        fprime = sij_inv_half @ hij @ sij_inv_half # core Hamiltonian as initial Fock matrix
        e, cprime = np.linalg.eigh(fprime)
        vecs = sij_inv_half @ cprime
        if abs(hij @ vecs - sij @ vecs @ np.diag(e)).max() > 1e-12: # FC=SCe
            raise ValueError('Generalized eigendecomposition failed.')
        Puv = 2 * vecs[:,:ne//2] @ vecs[:,:ne//2].T # 初始密度矩陣
    elif initial_guess.lower() == 'sap':
        veff = np.zeros((natom*nrad*nang,))
        for iatom in range(natom):
            symbol = mol.symbols[iatom]
            r, v = np.loadtxt(f'../data/sap/potentials/nr/LDAX/v_{symbol}.dat', dtype=float).T
            newr = dist[iatom] # (natomxnradxnang,)
            newv = np.interp(newr, r, v)
            veff += newv
        JXC = np.zeros((nao, nao))
        for mu in range(nao):
            for nu in range(mu,nao):
                JXC[mu,nu] = np.sum(aos_vals[mu].ravel() * veff * aos_vals[nu].ravel() * mweights.ravel())
                if nu > mu:
                    JXC[nu,mu] = JXC[mu,nu]
        fprime = sij_inv_half @ (hij + JXC) @ sij_inv_half
        e, cprime = np.linalg.eigh(fprime)
        vecs = sij_inv_half @ cprime
        if abs((hij + JXC) @ vecs - sij @ vecs @ np.diag(e)).max() > 1e-12:
            raise ValueError('Generalized eigendecomposition failed.')
        Puv = 2 * vecs[:,:ne//2] @ vecs[:,:ne//2].T # 初始密度矩陣
        

    # SCF
    print(f"# {'='*65} #")
    print(f"# {'='*30} SCF {'='*30} #")
    print(f"# {'='*65} #")
    while True:
        t0 = time.time()
        print(f'!>STEP {counter+1:<3d}')
        rho = np.zeros((natom, nrad*nang)) # 更新電子密度
        for iatom in range(natom):
            for mu in range(nao):
                for nu in range(mu,nao):
                    if nu > mu:
                        rho[iatom] += 2 * aos_vals[mu,iatom] * Puv[mu,nu] * aos_vals[nu,iatom]
                    else:
                        rho[iatom] += aos_vals[mu,iatom] * Puv[mu,nu] * aos_vals[nu,iatom]
        if abs(np.sum(rho * mweights) - ne) < 1e-3: # 人爲抹除數值誤差
            print(f"ne via quadrature {np.sum(rho * mweights):.15f}")
            rho *= ne / np.sum(rho * mweights)
        else:
            raise Exception

        # 構建全局庫倫勢  TODO: this is slow
        t1 = time.time()
        rho_ik = np.zeros((natom, nrad, nlm))
        u_ik = np.zeros((natom, nrad, nlm))
        u_ik_splines : list[CubicSpline] = [] # (natom,nlm)
        u_ik_interp = np.zeros(((natom, nlm, natom*nrad*nang)))
        u_ij = np.zeros((natom*nrad*nang,))
        for iatom in range(natom):
            qn = np.sum(rho[iatom] * mweights[iatom]) # ; print(f"    {iatom}, {qn}")
            rho_ik[iatom] = (becke.weight[iatom] * rho[iatom]).reshape((nrad, nang)) @ (y_jk * becke.wleb[:,None])
            u_ik_splines.append([])
            for ilm, (l,_) in enumerate(lm):
                u_ik[iatom,:,ilm] = poisson_solver(rho_ik[iatom,:,ilm], l, becke.rcheb[iatom], mol.rms[iatom], qn=qn)[1:-1]
                u_ik_splines[iatom].append(CubicSpline(becke.zcheb[iatom], u_ik[iatom,:,ilm]))
                u_ik_interp[iatom,ilm] = u_ik_splines[iatom][ilm](zz[iatom])
                u_ik_interp[iatom,ilm][mask_ubound[iatom]] = 0.
                if ilm == 0:
                    u_ik_interp[iatom,ilm][mask_lbound[iatom]] = np.sqrt(4. * np.pi) * qn
                else:
                    u_ik_interp[iatom,ilm][mask_lbound[iatom]] = 0.
            u_ij += np.sum((u_ik_interp[iatom] / dist[iatom]) * ylm[iatom], axis=0)
        t2 = time.time()


        # 計算算符矩陣 ⟨Φi|U|Φj⟩, ⟨Φi|K|Φj⟩, ⟨Φi|C|Φj⟩
        Juv  = np.zeros((nao, nao)) # 庫倫勢
        Kuv  = np.zeros((nao, nao)) # 交換勢
        Cuv  = np.zeros((nao, nao)) # 相關勢
        Exuv = np.zeros((nao, nao)) # 能量密度
        Ecuv = np.zeros((nao, nao)) # 能量密度
        
        inp = {'rho': rho.ravel()} # (natomxnradxnang, )
        ret_x = slaterX.compute(inp)
        ret_c = vwn5.compute(inp)
        vx = ret_x['vrho'].ravel() # potential. Otherwise, shape (ngrid, 1)
        vc = ret_c['vrho'].ravel() # potential
        ex = ret_x['zk'].ravel() # energy density
        ec = ret_c['zk'].ravel() # energy density

        for mu in range(nao):
            for nu in range(mu, nao):
                Juv[mu,nu]  = np.sum(aos_vals[mu].ravel() * u_ij * aos_vals[nu].ravel() * mweights.ravel())
                Kuv[mu,nu]  = np.sum(aos_vals[mu].ravel() * vx   * aos_vals[nu].ravel() * mweights.ravel())
                Cuv[mu,nu]  = np.sum(aos_vals[mu].ravel() * vc   * aos_vals[nu].ravel() * mweights.ravel())
                Exuv[mu,nu] = np.sum(aos_vals[mu].ravel() * ex   * aos_vals[nu].ravel() * mweights.ravel())
                Ecuv[mu,nu] = np.sum(aos_vals[mu].ravel() * ec   * aos_vals[nu].ravel() * mweights.ravel())
                if nu > mu:
                    Juv[nu,mu]  = Juv[mu,nu]
                    Kuv[nu,mu]  = Kuv[mu,nu]
                    Cuv[nu,mu]  = Cuv[mu,nu]
                    Exuv[nu,mu] = Exuv[mu,nu]
                    Ecuv[nu,mu] = Ecuv[mu,nu]
        tend = time.time()

        fock = tij + vij + Juv + Kuv + Cuv ; fock = (fock + fock.T) / 2 # 更新fock矩陣
        focks.append(fock)
        diis_res.append(sij_inv_half @ (fock @ Puv @ sij - sij @ Puv @ fock) @ sij_inv_half)


        # 能量分解
        kin_e         = np.trace(Puv @ tij.T)
        ext_e         = np.trace(Puv @ vij.T)
        hartree_e     = np.trace(Puv @ Juv.T) / 2
        exchange_e    = np.trace(Puv @ Exuv.T)
        correlation_e = np.trace(Puv @ Ecuv.T)
        etot = kin_e + ext_e + hartree_e + exchange_e + correlation_e + mol.e_nuc
        print(f"Total wall time                          {tend-t0:>20.15f} sec")
        print(f"Time for build up electron density       {t1-t0:>20.15f} sec")
        print(f"Time for build up Coulomb potential      {t2-t1:>20.15f} sec")
        print(f"Time for build up Fock matrix            {tend-t2:>20.15f} sec")
        print(f"Number of electrons via quadrature       {np.sum(rho * mweights):>20.15f}")
        print(f'One-electron energy                      {kin_e + ext_e:>20.15f}')
        print(f'Two-electron energy                      {hartree_e:>20.15f}')
        print(f'Exchange-correlation energy              {exchange_e + correlation_e:>20.15f}  {exchange_e}  {correlation_e}')
        print(f'Total energy                             {etot:>20.15f}')
        print(f'Energy difference                        {etot - etot_old:>20.15f}')


        # DIIS
        # - Chem Phys Lett 73(2):393-398
        # - J Comput Chem 3(4):556-560
        # - https://doi.org/10.1007/s00214-018-2238-8
        # - https://github.com/psi4/psi4numpy/blob/master/Tutorials/03_Hartree-Fock/3b_rhf-diis.ipynb
        if counter > 2:
            B = np.zeros((len(focks)+1, len(focks)+1))
            B[-1,:] = -1.
            B[:,-1] = -1.
            B[-1,-1] = 0.

            for ii in range(len(focks)):
                for jj in range(len(focks)):
                    B[ii,jj] = np.trace(diis_res[ii] @ diis_res[jj].T)
            diis_rhs = np.zeros((len(B),))
            diis_rhs[-1] = -1.
            diis_coef = np.linalg.solve(B, diis_rhs)

            fock = np.zeros(fock.shape)
            for iii in range(len(diis_coef)-1):
                fock += focks[iii] * diis_coef[iii]
            np.testing.assert_allclose(fock, fock.T, atol=1e-15)
            fock = (fock + fock.T) / 2

        # 對新的fock矩陣對角化
        fprime = sij_inv_half @ fock @ sij_inv_half
        e, cprime = np.linalg.eigh(fprime)
        vecs = sij_inv_half @ cprime
        if abs(fock @ vecs - sij @ vecs @ np.diag(e)).max() > 1e-12: # FC=SCe
            raise Exception
        Puv = 2 * vecs[:,:ne//2] @ vecs[:,:ne//2].T # 更新密度矩陣

        # 收斂判斷
        if abs(etot - etot_old) < e_convergence and np.sqrt(np.sum((Puv - Puv_old)**2)) < d_convergence:
            print(f'RMSD                                     {np.sqrt(np.sum((Puv - Puv_old)**2)):>15.8f}')
            print()
            print("!> NORMAL TERMINATION !<\n")
            print(f'SCF converged after {counter+1} step. Total energy {etot:<15.8f}')
            print(f"Occupied orbital energies: {e[:ne//2]}")
            print(f"Unoccupied orbital energies: {e[ne//2:]}")
            break

        if counter >= maxiter:
            print(f'SCF did not converge. Energy in the last iteration {etot:<15.8f}')
            break

        print(f'RMSD                                     {np.sqrt(np.sum((Puv - Puv_old)**2) / nao**2):>15.8f}')
        print()

        counter += 1
        etot_old = etot
        Puv_old = Puv

        
        
    
if __name__ == '__main__':
    dft(xyzfile='../data/xyz/ch4.xyz', basefile='sto-3g', initial_guess='core', d_convergence=1e-6, e_convergence=1e-6)