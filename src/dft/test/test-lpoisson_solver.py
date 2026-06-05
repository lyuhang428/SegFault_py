import numpy as np
from lbeckeGrid import poisson_solver, gaussCheby2
from lrsh import *
from scipy.special import gamma
from scipy.integrate import lebedev_rule

def test_lpoisson_solver_gaussCheby2():
    f = lambda x: x**2 * np.exp(-0.5 * x**2) # \int_0^\infty f^2 dx = 3 \sqrt{\pi} / 8 
    val = 0.6646701940895685
    _, rcheb, wcheb = gaussCheby2()
    integral = np.sum(f(rcheb)**2 * wcheb)
    print(f'Absolute error: {integral-val:>20.16f}') # Absolute error:  -0.0000002505416914

def test_lpoisson_solver_poisson_solver(): 
    '''test Fe sto-3g dxy orbital'''
    EXPNS = np.array([0.6411803475E+01, 0.1955804428E+01, 0.7546101508E+00])
    COEFS = np.array([0.2197679508E+00, 0.6555473627E+00, 0.2865732590E+00])
    LANG = 2 # angular momentum quantum number l for d orbital
    NRFCS = np.sqrt(2. * (2 * EXPNS)**(LANG + 1.5) / gamma(LANG + 1.5))

    rxy = lambda r: NRFCS[0] * COEFS[0] * r**LANG * np.exp(-EXPNS[0] * r**2) + NRFCS[1] * COEFS[1] * r**LANG * np.exp(-EXPNS[1] * r**2) + NRFCS[2] * COEFS[2] * r**LANG * np.exp(-EXPNS[2] * r**2)
    psi = lambda r, x, y, z: Y2m2(x, y, z) * rxy(r)

    # Gauss-Chebyshev-Lebedev grid
    n = 32
    rm = 2.6456
    _, rcheb, wcheb = gaussCheby2(n, rm)
    xleb, wleb = lebedev_rule(17)

    olp = np.sum([wi * ri**2 * (psi(ri, *xleb)**2 * wleb).sum() for ri, wi in zip(rcheb, wcheb)])
    np.testing.assert_allclose(olp, 1., )
    print(olp)

    # solve Poisson equation from rho_lm to get u_lm
    rho44 = np.array([np.sum(Y44(*xleb) * psi(ri, *xleb)**2 * wleb) for ri in rcheb])
    rho00 = np.array([np.sum(Y00(*xleb) * psi(ri, *xleb)**2 * wleb) for ri in rcheb])
    rho20 = np.array([np.sum(Y20(*xleb) * psi(ri, *xleb)**2 * wleb) for ri in rcheb])
    rho40 = np.array([np.sum(Y40(*xleb) * psi(ri, *xleb)**2 * wleb) for ri in rcheb])

    u44 = poisson_solver(rho44, 4, rm)
    u00 = poisson_solver(rho00, 0, rm)
    u20 = poisson_solver(rho20, 2, rm)
    u40 = poisson_solver(rho40, 4, rm)
    eri44 = np.sum(rcheb * rho44 * u44[1:-1] * wcheb)
    eri00 = np.sum(rcheb * rho00 * u00[1:-1] * wcheb)
    eri20 = np.sum(rcheb * rho20 * u20[1:-1] * wcheb)
    eri40 = np.sum(rcheb * rho40 * u40[1:-1] * wcheb)

    print(eri00) # eri00 = 0.967496
    print(eri20) # eri20 = 0.041713
    print(eri40) # eri40 = 0.000755
    print(eri44) # eri44 = 0.026435


def test_contraction_notion():
    '''test Fe sto-3g d orbital'''
    EXPNS = np.array([0.6411803475E+01, 0.1955804428E+01, 0.7546101508E+00])
    COEFS = np.array([0.2197679508E+00, 0.6555473627E+00, 0.2865732590E+00])
    LANG = 2 
    NRFCS = np.sqrt(2. * (2 * EXPNS)**(LANG + 1.5) / gamma(LANG + 1.5))

    rxy = lambda r: NRFCS[0] * COEFS[0] * r**LANG * np.exp(-EXPNS[0] * r**2) + NRFCS[1] * COEFS[1] * r**LANG * np.exp(-EXPNS[1] * r**2) + NRFCS[2] * COEFS[2] * r**LANG * np.exp(-EXPNS[2] * r**2)
    psi = lambda r, x, y, z: Y2m2(x, y, z) * rxy(r) # psi_dxy ao of Fe at sto-3g


    ncheb = 32
    nleb = 17
    rm = 2.6456
    _, rcheb, wcheb = gaussCheby2(ncheb, rm)
    xleb, wleb = lebedev_rule(nleb)

    # 重叠积分基准测试
    olp = np.sum([wi * ri**2 * np.sum(psi(ri, *xleb)**2 * wleb) for wi, ri in zip(wcheb, rcheb)])
    # print(olp)


    # 球谐展开
    # 对于d原子轨道，径向和角度部分可以分离，所以球谐函数实际上是对|Y_2|^2进行展开. 这是一个解析问题，由Clebsch-Gordan系数
    # 以及角动量代数可以解析求得展开系数，且展开项数有限，同时只有满足特定要求的l,m的系数不为零，这就是为什么只有在l=0,2,4时
    # 满足特定要求的m的项的rho_lm不为零
    lm = [(0, 0), 
          (1,-1), (1, 0), (1, 1),
          (2,-2), (2,-1), (2, 0), (2, 1), (2,2), 
          (3,-3), (3,-2), (3,-1), (3, 0), (3,1), (3,2), (3,3), 
          (4,-4), (4,-3), (4,-2), (4,-1), (4,0), (4,1), (4,2), (4,3), (4,4)]
    rho_ik = np.zeros((    ncheb, len(lm))) # 这里虽然记作rho，但是并非电子密度，对原子轨道谈论电子密度没有意义. 这里单纯是重叠积分
    y_jk   = np.zeros((len(wleb), len(lm)))
    for i, (l, m) in enumerate(lm):
        y_jk[:,i]   = RSH[f'{l}'][f'{m}'](*xleb)
        rho_ik[:,i] = [np.sum(y_jk[:,i] * (psi(ri, *xleb))**2 * wleb) for ri in rcheb]
        print(f'l={l}\tm={m}\tmax={abs(rho_ik[:,i]).max()}')

    # 显示循环
    olp = 0.
    for i in range(ncheb):
        tmp = 0.
        for k in range(25):
            tmp += rho_ik[i,k] * np.sum(y_jk[:,k] * wleb)
        olp += tmp * rcheb[i]**2 * wcheb[i]
    print(olp)
    

    # 爱因斯坦求和
    olp = np.einsum('i,i,ik,j,jk->', wcheb, rcheb**2, rho_ik, wleb, y_jk)
    print(olp)


    # print(rho_ik.shape, y_jk.shape)      # y_jk单纯是不同lm球谐函数在球面采样点处的离散函数值按列拼成的矩阵
    # rho_ij = rho_ik @ y_jk.T             # rho_ik是一种球谐展开，rho_ij是在第i个径向点处第j个球面点处电子密度
    # print(rho_ij.shape)
    # tmp = rho_ij @ (y_jk * wleb[:,None]) # 实际上也是一种球谐展开
    # print(tmp.shape)
    # print(np.allclose(tmp, rho_ik))





if __name__ == '__main__':
    # test_lpoisson_solver_poisson_solver()
    test_contraction_notion()