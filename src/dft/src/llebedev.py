import numpy as np

LEBEDEV_ORDER = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41, 47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131, ]
LEBEDEV_LEVEL = [6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350, 434, 590, 770, 974, 1202, 1454, 1730, 2030, 2354, 2702, 3074, 3470, 3890, 4334, 4802, 5294, 5810, ]

def lebedev_rule(n:int) -> tuple[np.ndarray, np.ndarray]:
    '''
    Argument
    --------
    n : int
        order of lebedev point

    Return
    ------
    (xleb, wleb) : tuple[np.ndarray, np.ndarray]
        xleb (3, nang), rows are x, y, z coordinates on unit sphere
        
        wleb (nang, ), sampling point weight
    '''

    if n not in LEBEDEV_ORDER:
        raise ValueError(f"Allowed Lebedev orders are {LEBEDEV_ORDER}")
    n = str(n).rjust(3, '0')
    data = np.loadtxt(f'../data/lebedev/lebedev_{n}.dat', dtype=float)
    theta = data[:,1] / 180. * np.pi
    phi = data[:,0] / 180. * np.pi
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    w = data[:,2] * 4 * np.pi

    return np.stack((x,y,z), axis=0), w


if __name__ == '__main__':
    xyz, w = lebedev_rule(17)
    xwleb = np.vstack((xyz, w))
    for ii in xwleb.T:
        print(ii)
