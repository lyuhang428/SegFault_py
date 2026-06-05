import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.integrate import lebedev_rule
from scipy.special import gamma
from scipy.interpolate import griddata
from sys import exit
from rsp import Y00, Y20, Y40, Y44, Y2m2


'''Fe sto-3g
S    3   1.00
      0.1447400411E+04       0.1543289673E+00
      0.2636457916E+03       0.5353281423E+00
      0.7135284019E+02       0.4446345422E+00
SP   3   1.00
      0.1119194891E+03      -0.9996722919E-01       0.1559162750E+00
      0.2600768236E+02       0.3995128261E+00       0.6076837186E+00
      0.8458505490E+01       0.7001154689E+00       0.3919573931E+00
SP   3   1.00
      0.8548569754E+01      -0.2277635023E+00       0.4951511155E-02
      0.2607586250E+01       0.2175436044E+00       0.5777664691E+00
      0.1006087840E+01       0.9166769611E+00       0.4846460366E+00
SP   3   1.00
      0.5921156814E+00      -0.3088441214E+00      -0.1215468600E+00
      0.2185279254E+00       0.1960641165E-01       0.5715227604E+00
      0.9650423590E-01       0.1131034442E+01       0.5498949471E+00
D    3   1.00
      0.6411803475E+01       0.2197679508E+00
      0.1955804428E+01       0.6555473627E+00
      0.7546101508E+00       0.2865732590E+00
'''

EXPNS = np.array([0.6411803475E+01, 0.1955804428E+01, 0.7546101508E+00])
COEFS = np.array([0.2197679508E+00, 0.6555473627E+00, 0.2865732590E+00])
LANG = 2 # angular momentum quantum number l for d orbital
NRFCS = np.sqrt(2. * (2 * EXPNS)**(LANG + 1.5) / gamma(LANG + 1.5))

rxy = lambda r: NRFCS[0] * COEFS[0] * r**LANG * np.exp(-EXPNS[0] * r**2) + NRFCS[1] * COEFS[1] * r**LANG * np.exp(-EXPNS[1] * r**2) + NRFCS[2] * COEFS[2] * r**LANG * np.exp(-EXPNS[2] * r**2)
psi = lambda r, x, y, z: Y2m2(x, y, z) * rxy(r)


def fe3dxy_plot():
    xi = np.linspace(-2.5, 2.5, 50)
    yi = np.linspace(-2.5, 2.5, 50)
    zi = np.linspace(-2.5, 2.5, 50)
    x, y, z = np.meshgrid(xi, yi, zi)
    psi = 1 / 2 * np.sqrt(15 / np.pi) * x * y / np.sqrt(x**2 + y**2 + z**2) * rxy(np.sqrt(x**2 + y**2 + z**2))
    
    fig = plt.subplots(figsize=(4, 4))
    ax = fig.add_subplots(111, projection='3d')
    # ax.contour(x, y, psi, 10)
    ax.contour(x, y, z, psi, )
    ax.set_aspect('equal')
    ax.set_xticks([-2.5, -2, -1, 0, 1., 2., 2.5])
    ax.set_yticks([-2.5, -2, -1, 0, 1., 2., 2.5])
    ax.set_xticklabels(['', '-2', '-1', '0', '1', '2', ''])
    ax.set_yticklabels(['', '-2', '-1', '0', '1', '2', ''])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # ax.grid(linewidth=0.5, linestyle='--', color='grey')
    plt.tight_layout()
    plt.show()




def gaussCheby2(norder:int=128, rm:float=1.) -> tuple[np.ndarray, np.ndarray]:
    x = np.cos(np.arange(1, norder+1) / (norder + 1) * np.pi) # z ∈ [1, norder] ↦ x ∈ (1, -1)
    r = rm * (1 + x) / (1 - x)                                # r ∈ [0, ∞)
    w = np.pi / (norder + 1) * np.sin(np.arange(1, norder+1) / (norder + 1) * np.pi)**2 / np.sqrt(1 - x**2) * 2 * rm / (1 - x)**2
    return x, r, w




def olp():
    '''测试Fe 3dxy 原子轨道的重叠积分是否归一. 测试归一化系数和波函数的函数形式是否正确'''
    xleb, wleb = lebedev_rule(17)

    norder = 128
    rm = 2.6456
    xcheb, rcheb, wcheb = gaussCheby2(norder, rm)

    integral = 0.
    for ri, wi in zip(rcheb, wcheb):
        integral += np.sum(psi(ri, *xleb)**2 * ri**2 * wleb) * wi
    
    print(f'integral = {integral}')



def rho_lm():
    xleb, wleb = lebedev_rule(17)

    norder = 24
    rm = 2.6456
    xcheb, rcheb, wcheb = gaussCheby2(norder, rm)

    rho00 = np.array([np.sum(Y00(*xleb) * psi(ri, *xleb)**2 * wleb) for ri in rcheb]) # rho是位矢\vec{r}的函数，rho_{lm}是积掉固体角后的电子密度，是标量r的函数
    rho20 = np.array([np.sum(Y20(*xleb) * psi(ri, *xleb)**2 * wleb) for ri in rcheb]) # rho_{lm} 是球谐函数展开系数
    rho40 = np.array([np.sum(Y40(*xleb) * psi(ri, *xleb)**2 * wleb) for ri in rcheb]) # rho拆分成径向部分rho_{lm}(r)和角度部分Y_{lm}(\theta, \phi)的线性组合
    rho44 = np.array([np.sum(Y44(*xleb) * psi(ri, *xleb)**2 * wleb) for ri in rcheb]) # 所以rho_{lm}事实上是展开系数

    fig, ax = plt.subplots(figsize=(6,2))
    ax.semilogx(rcheb, rho00, 'o', markersize=6, c='#40925bff', label=r'$Y_{00}$', markeredgecolor='black')
    ax.semilogx(rcheb, rho20, 'o', markersize=6, c='#69bbacff', label=r'$Y_{20}$', markeredgecolor='black')
    ax.semilogx(rcheb, rho44, 'o', markersize=6, c='#adad5bff', label=r'$Y_{44}$', markeredgecolor='black')
    ax.semilogx(rcheb, rho40, 'o', markersize=6, c='#e3d69bff', label=r'$Y_{40}$', markeredgecolor='black')
    ax.set_xlim((1e-3,1e3))
    ax.set_ylim((-1, 1))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()



def fig4_25():
    l0 = 0
    l2 = 2
    l4 = 4

    '''切比雪夫，列别杰夫网格'''
    n  = 128
    rm = 2.6456
    xcheb, rcheb, wcheb = gaussCheby2(n, rm)
    xleb, wleb = lebedev_rule(17)

    '''电子密度，电子径向密度'''
    rho00r2 = np.array([np.sum(Y00(*xleb) * psi(ri, *xleb)**2 * ri**2 * wleb) for ri in rcheb])
    rho00   = np.array([np.sum(Y00(*xleb) * psi(ri, *xleb)**2 * wleb) for ri in rcheb])

    rho20r2 = np.array([np.sum(Y20(*xleb) * psi(ri, *xleb)**2 * ri**2 * wleb) for ri in rcheb])
    rho20   = np.array([np.sum(Y20(*xleb) * psi(ri, *xleb)**2 * wleb) for ri in rcheb])

    rho40r2 = np.array([np.sum(Y40(*xleb) * psi(ri, *xleb)**2 * ri**2 * wleb) for ri in rcheb])
    rho40   = np.array([np.sum(Y40(*xleb) * psi(ri, *xleb)**2 * wleb) for ri in rcheb])

    rho44r2 = np.array([np.sum(Y44(*xleb) * psi(ri, *xleb)**2 * ri**2 * wleb) for ri in rcheb])
    rho44   = np.array([np.sum(Y44(*xleb) * psi(ri, *xleb)**2 * wleb) for ri in rcheb])


    '''有限差分求解电子泊松方程计算电子库伦势'''
    dzdr2  = (n + 1)**2 * rm / (np.pi**2 * rcheb) / (rm + rcheb)**2
    d2zdr2 = (n + 1) * rm**2 * (3 * rcheb + rm) / (2 * np.pi) / (rcheb * rm)**1.5 / (rcheb + rm)**2

    D1   = np.zeros((n+2, n+2))
    D2   = np.zeros((n+2, n+2))
    
    b00 = np.zeros((n+2,))
    b20 = np.zeros((n+2,))
    b40 = np.zeros((n+2,))
    b44 = np.zeros((n+2,))

    diag00 = np.diag(l0 * (l0 + 1) / rcheb**2)
    diag20 = np.diag(l2 * (l2 + 1) / rcheb**2)
    diag40 = np.diag(l4 * (l4 + 1) / rcheb**2)
    diag44 = np.diag(l4 * (l4 + 1) / rcheb**2)
 
    D1[  1, 0:5]   = np.array([-3, -10, 18, -6, 1]) / 12
    D1[  2, 0:6]   = np.array([3, -30, -20, 60, -15, 2]) / 60
    D1[n-1, -6:]   = np.array([-2, 15, -60, 20, 30, -3]) / 60
    D1[  n, -5:]   = np.array([-1, 6, -18, 10, 3]) / 12
    kernel_7d1     = np.array([-1, 9, -45, 0., 45, -9, 1]) / 60
    for i in range(3, n-1):
        D1[i, i-3:i+4] = kernel_7d1
    D1[1:-1] *= d2zdr2[:,None]

    D2[  1, 0:5]   = np.array([11, -20, 6, 4, -1]) / 12
    D2[  2, 0:5]   = np.array([-1, 16, -30, 16, -1]) / 12
    D2[n-1, -5:]   = np.array([-1, 16, -30, 16, -1]) / 12
    D2[  n, -5:]   = np.array([-1, 4, 6, -20, 11]) / 12
    kernel_7d2     = np.array([2, -27, 270, -490, 270, -27, 2]) / 180
    for i in range(3, n-1):
        D2[i, i-3:i+4] = kernel_7d2
    D2[1:-1] *= dzdr2[:,None]

    A00 = D2 + D1
    A00[1:-1,1:-1] -= diag00
    A00[0, 0] = 1.
    A00[-1,-1] = 1.

    A20 = D2 + D1
    A20[1:-1,1:-1] -= diag20
    A20[0, 0] = 1.
    A20[-1,-1] = 1.

    A40 = D2 + D1
    A40[1:-1,1:-1] -= diag40
    A40[0, 0] = 1.
    A40[-1,-1] = 1.

    A44 = D2 + D1
    A44[1:-1,1:-1] -= diag44
    A44[0, 0] = 1.
    A44[-1,-1] = 1.

    b00[1:-1] = -4 * np.pi * rcheb * rho00
    b00[0] = np.sqrt(4 * np.pi)
    b00[-1] = 0

    b20[1:-1] = -4 * np.pi * rcheb * rho20
    b20[0] = np.sqrt(4 * np.pi)
    b20[-1] = 0

    b40[1:-1] = -4 * np.pi * rcheb * rho40
    b40[0] = np.sqrt(4 * np.pi)
    b40[-1] = 0

    b44[1:-1] = -4 * np.pi * rcheb * rho44
    b44[0] = np.sqrt(4 * np.pi)
    b44[-1] = 0
    
    u00 = np.linalg.inv(A00) @ b00
    u20 = np.linalg.inv(A20) @ b20
    u40 = np.linalg.inv(A40) @ b40
    u44 = np.linalg.inv(A44) @ b44
    
    '''电子自排斥积分 [ii|ii]'''
    eri00 = np.sum((rcheb * rho00 * u00[1:-1]) * wcheb)
    eri20 = np.sum((rcheb * rho20 * u20[1:-1]) * wcheb)
    eri40 = np.sum((rcheb * rho40 * u40[1:-1]) * wcheb)
    eri44 = np.sum((rcheb * rho44 * u44[1:-1]) * wcheb)
    print(f'eri00 = {eri00:>10.6f}')
    print(f'eri20 = {eri20:>10.6f}')
    print(f'eri40 = {eri40:>10.6f}')
    print(f'eri44 = {eri44:>10.6f}')
    print(eri00 + eri20 + eri40 + eri44)


    exit()
    '''绘图'''
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot()
    ax.semilogx(rcheb, rho00r2, 'o', markersize=6, c='#40925bff', markeredgecolor='black', label=r'$\rho_{00}$')
    ax.semilogx(rcheb, rho20r2, 'o', markersize=6, c='#69bbacff', markeredgecolor='black', label=r'$\rho_{20}$')
    ax.semilogx(rcheb, rho40r2, 'o', markersize=6, c='#adad5bff', markeredgecolor='black', label=r'$\rho_{40}$')
    ax.semilogx(rcheb, rho44r2, 'o', markersize=6, c='#e3d69bff', markeredgecolor='black', label=r'$\rho_{44}$')
    ax.set_xlim((1e-3, 1e3))
    ax.set_ylim((-0.4, 1.))
    ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(['', '-0.2', '0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_ylabel(r'$r^2 \rho_{lm}(r)$')
    ax.legend(loc='upper left')
    ax.grid(color='grey', linestyle='--', linewidth=0.5)

    ax1 = ax.twinx()
    ax1.semilogx(rcheb, u00[1:-1] / rcheb, '--', c='#69bbadff', label=r'$r^{-1}U_{00}$')
    ax1.semilogx(rcheb, u20[1:-1] / rcheb, '--', c='#40925bff', label=r'$r^{-1}U_{20}$')
    ax1.semilogx(rcheb, u40[1:-1] / rcheb, '--', c='#adad5bff', label=r'$r^{-1}U_{40}$')
    ax1.semilogx(rcheb, u44[1:-1] / rcheb, '--', c='#e3d692ff', label=r'$r^{-1}U_{44}$')
    ax1.set_ylim((-2, 5))
    ax1.set_yticks([-2, -1, 0, 1, 2, 3, 4, 5])
    ax1.set_yticklabels(['', '-1', '0', '1', '2', '3', '4', '5'])
    ax1.set_ylabel(r'$r^{-1} U_{lm}(r)$')
    ax1.legend(loc='upper right')

    plt.tight_layout()
    plt.show()




if __name__ == '__main__':
    # rho_lm()
    # fig4_25()
    # olp()
    # fe3dxy_plot()
    pass
