import time
import numpy as np
from scipy.special import sph_harm_y, sph_harm_y_all

def timeit(f:callable) -> callable:
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = f(*args, **kwargs)
        print(f'{f.__name__} took {time.time()-t0:<15.8f} sec')
        return result
    return wrapper

# @timeit
def rsh(l:int, m:int, theta:float, phi:float) -> float:
    '''
    Reference
    ---------
    - https://scipython.com/blog/visualizing-the-real-forms-of-the-spherical-harmonics/ 
    - https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    '''
    if abs(m) > l:
        raise ValueError('magnetic quantum number m should not be larger than angular qunautm number l')
    
    # phase = 1 if (abs(m) % 2 == 0) else -1 # (-1)^m
    phase = -1 if abs(m) % 2 else 1

    if m > 0:
        # return phase * np.sqrt(2.) * sph_harm_y(l, m, theta, phi).real
        return np.sqrt(2.) * sph_harm_y(l, m, theta, phi).real
    elif m < 0:
        # return phase * np.sqrt(2.) * sph_harm_y(l, -m, theta, phi).imag
        return np.sqrt(2.) * sph_harm_y(l, -m, theta, phi).imag
    elif m == 0:
        return sph_harm_y(l, 0, theta, phi).real # a+0j, forcefully remove null imaginary part


if __name__ == '__main__':
    ix = np.linspace(1, 10, 40)
    iy = np.linspace(1, 10, 40)
    iz = np.linspace(1, 10, 40)
    x, y, z = np.meshgrid(ix,iy,iz)
    xyz = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
    dist = np.linalg.norm(xyz, axis=1)
    xyz_normalized = xyz / dist[:,None] ; np.testing.assert_allclose(np.linalg.norm(xyz_normalized, axis=1), 1.)
    theta = np.arccos(xyz_normalized[:,2])
    phi = np.arctan2(xyz_normalized[:,1], xyz_normalized[:,0])
    ngrid = len(xyz_normalized)
    lmax = 14
    nlm = (lmax + 1)**2


    tmp = np.stack((sph_harm_y(1,0,theta,phi), sph_harm_y(1,-1,theta,phi), sph_harm_y(1,1,theta,phi)), axis=0)
    tmp2 = sph_harm_y_all(1,1,theta,phi)[1]
    np.testing.assert_allclose(sph_harm_y(1,1,theta,phi), sph_harm_y_all(1,1,theta,phi)[-1,-1].ravel())