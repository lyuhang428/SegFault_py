# References
# https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
import numpy as np
from scipy.special import sph_harm_y


def __spherical_harmonic(l, m, theta, phi):
    """
    Calculate value of spherical harmonic function. Copied from PyDFT: [https://github.com/ifilot/pydft]. 
    
    l:     angular quantum number
    m:     magnetic quantum number
    theta: azimuthal angle in radians
    phi:   polar angle in radians
    """
    # no Condon-Shortley phase
    if m < 0:
        val = np.sqrt(2) * np.imag(sph_harm_y(l, np.abs(m), phi, theta))
    elif m > 0:
        val = np.sqrt(2) * np.real(sph_harm_y(l, m, phi, theta))
    else:
        val = np.real(sph_harm_y(l, m, phi, theta))
    
    return val

def __spherical_harmonic_cart(l, m, p):
    """
    Calculate the value of the spherical harmonic depending on the position
    p in Cartesian coordinates. This function assumes that the position p
    lies on the unit sphere. Copied from PyDFT: [https://github.com/ifilot/pydft]. 
    
    l:    angular quantum number
    m:    magnetic quantum number
    p:    position three-vector on the unit sphere
    """
    theta = np.arctan2(p[1], p[0])  # azimuthal
    phi = np.arccos(p[2])           # polar

    return __spherical_harmonic(l, m, theta, phi)

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
        return phase * np.sqrt(2.) * sph_harm_y(l, m, theta, phi).real
    elif m < 0:
        return phase * np.sqrt(2.) * sph_harm_y(l, -m, theta, phi).imag
    elif m == 0:
        return sph_harm_y(l, 0, theta, phi).real # a+0j, forcefully remove null imaginary part

def rsh2(l:int, m:int, theta:float, phi:float) -> float:
    '''
    Reference
    ---------
    - https://docs.abinit.org/theory/spherical_harmonics/
    '''
    if abs(m) > l:
        raise ValueError('magnetic quantum number m should not be larger than angular qunautm number l')
    
    phase = 1 if (abs(m) % 2 == 0) else -1 # (-1)^m

    if m > 0:
        return phase / np.sqrt(2.) * (sph_harm_y(l, m, theta, phi) + sph_harm_y(l, m, theta, phi).conjugate())
    elif m < 0:
        return phase / np.sqrt(2.) / 1j * (sph_harm_y(l, -m, theta, phi) - sph_harm_y(l, -m, theta, phi).conjugate())
    elif m == 0:
        return sph_harm_y(l, 0, theta, phi)


r'''\sqrt{\frac{2 * (2 * \alpha)^(l + 3/2)}{\Gamma(l + 3/2)}}'''
RSH_SPH = {
    '0': {
        '0': lambda t, p: np.sqrt(1 / 4 / np.pi) + t + p - t - p
    }, 
    '1': {
        '-1': lambda t, p: __spherical_harmonic(l=1, m=-1, theta=p, phi=t), 
         '0': lambda t, p: __spherical_harmonic(l=1, m= 0, theta=p, phi=t), 
         '1': lambda t, p: __spherical_harmonic(l=1, m= 1, theta=p, phi=t), 
    }, 
    '2': {
        '-2': lambda t, p: __spherical_harmonic(l=2, m=-2, theta=p, phi=t), 
        '-1': lambda t, p: __spherical_harmonic(l=2, m=-1, theta=p, phi=t), 
         '0': lambda t, p: __spherical_harmonic(l=2, m= 0, theta=p, phi=t), 
         '1': lambda t, p: __spherical_harmonic(l=2, m= 1, theta=p, phi=t), 
         '2': lambda t, p: __spherical_harmonic(l=2, m= 2, theta=p, phi=t),         
    }, 
    '3': {
        '-3': lambda t, p: __spherical_harmonic(l=3, m=-3, theta=p, phi=t), 
        '-2': lambda t, p: __spherical_harmonic(l=3, m=-2, theta=p, phi=t), 
        '-1': lambda t, p: __spherical_harmonic(l=3, m=-1, theta=p, phi=t), 
         '0': lambda t, p: __spherical_harmonic(l=3, m= 0, theta=p, phi=t), 
         '1': lambda t, p: __spherical_harmonic(l=3, m= 1, theta=p, phi=t), 
         '2': lambda t, p: __spherical_harmonic(l=3, m= 2, theta=p, phi=t), 
         '3': lambda t, p: __spherical_harmonic(l=3, m= 3, theta=p, phi=t), 
    }, 
    '4': {
        '-4': lambda t, p: __spherical_harmonic(l=4, m=-4, theta=p, phi=t), 
        '-3': lambda t, p: __spherical_harmonic(l=4, m=-3, theta=p, phi=t), 
        '-2': lambda t, p: __spherical_harmonic(l=4, m=-2, theta=p, phi=t), 
        '-1': lambda t, p: __spherical_harmonic(l=4, m=-1, theta=p, phi=t), 
         '0': lambda t, p: __spherical_harmonic(l=4, m= 0, theta=p, phi=t), 
         '1': lambda t, p: __spherical_harmonic(l=4, m= 1, theta=p, phi=t), 
         '2': lambda t, p: __spherical_harmonic(l=4, m= 2, theta=p, phi=t), 
         '3': lambda t, p: __spherical_harmonic(l=4, m= 3, theta=p, phi=t), 
         '4': lambda t, p: __spherical_harmonic(l=4, m= 4, theta=p, phi=t), 
    }, 
    '5': {
        '-5': lambda t, p: __spherical_harmonic(l=5, m=-5, theta=p, phi=t), 
        '-4': lambda t, p: __spherical_harmonic(l=5, m=-4, theta=p, phi=t), 
        '-3': lambda t, p: __spherical_harmonic(l=5, m=-3, theta=p, phi=t), 
        '-2': lambda t, p: __spherical_harmonic(l=5, m=-2, theta=p, phi=t), 
        '-1': lambda t, p: __spherical_harmonic(l=5, m=-1, theta=p, phi=t), 
         '0': lambda t, p: __spherical_harmonic(l=5, m= 0, theta=p, phi=t), 
         '1': lambda t, p: __spherical_harmonic(l=5, m= 1, theta=p, phi=t), 
         '2': lambda t, p: __spherical_harmonic(l=5, m= 2, theta=p, phi=t), 
         '3': lambda t, p: __spherical_harmonic(l=5, m= 3, theta=p, phi=t), 
         '4': lambda t, p: __spherical_harmonic(l=5, m= 4, theta=p, phi=t), 
         '5': lambda t, p: __spherical_harmonic(l=5, m= 5, theta=p, phi=t), 
    }, 
    '6': {
        '-6': lambda t, p: __spherical_harmonic(l=6, m=-6, theta=p, phi=t), 
        '-5': lambda t, p: __spherical_harmonic(l=6, m=-5, theta=p, phi=t), 
        '-4': lambda t, p: __spherical_harmonic(l=6, m=-4, theta=p, phi=t), 
        '-3': lambda t, p: __spherical_harmonic(l=6, m=-3, theta=p, phi=t), 
        '-2': lambda t, p: __spherical_harmonic(l=6, m=-2, theta=p, phi=t), 
        '-1': lambda t, p: __spherical_harmonic(l=6, m=-1, theta=p, phi=t), 
         '0': lambda t, p: __spherical_harmonic(l=6, m= 0, theta=p, phi=t), 
         '1': lambda t, p: __spherical_harmonic(l=6, m= 1, theta=p, phi=t), 
         '2': lambda t, p: __spherical_harmonic(l=6, m= 2, theta=p, phi=t), 
         '3': lambda t, p: __spherical_harmonic(l=6, m= 3, theta=p, phi=t), 
         '4': lambda t, p: __spherical_harmonic(l=6, m= 4, theta=p, phi=t), 
         '5': lambda t, p: __spherical_harmonic(l=6, m= 5, theta=p, phi=t), 
         '6': lambda t, p: __spherical_harmonic(l=6, m= 6, theta=p, phi=t), 
    }, 
    '7': {
        '-7': lambda t, p: __spherical_harmonic(l=7, m=-7, theta=p, phi=t), 
        '-6': lambda t, p: __spherical_harmonic(l=7, m=-6, theta=p, phi=t), 
        '-5': lambda t, p: __spherical_harmonic(l=7, m=-5, theta=p, phi=t), 
        '-4': lambda t, p: __spherical_harmonic(l=7, m=-4, theta=p, phi=t), 
        '-3': lambda t, p: __spherical_harmonic(l=7, m=-3, theta=p, phi=t), 
        '-2': lambda t, p: __spherical_harmonic(l=7, m=-2, theta=p, phi=t), 
        '-1': lambda t, p: __spherical_harmonic(l=7, m=-1, theta=p, phi=t), 
         '0': lambda t, p: __spherical_harmonic(l=7, m= 0, theta=p, phi=t), 
         '1': lambda t, p: __spherical_harmonic(l=7, m= 1, theta=p, phi=t), 
         '2': lambda t, p: __spherical_harmonic(l=7, m= 2, theta=p, phi=t), 
         '3': lambda t, p: __spherical_harmonic(l=7, m= 3, theta=p, phi=t), 
         '4': lambda t, p: __spherical_harmonic(l=7, m= 4, theta=p, phi=t), 
         '5': lambda t, p: __spherical_harmonic(l=7, m= 5, theta=p, phi=t), 
         '6': lambda t, p: __spherical_harmonic(l=7, m= 6, theta=p, phi=t), 
         '7': lambda t, p: __spherical_harmonic(l=7, m= 7, theta=p, phi=t), 
    }, 
    '8': {
        '-8': lambda t, p: __spherical_harmonic(l=8, m=-8, theta=p, phi=t), 
        '-7': lambda t, p: __spherical_harmonic(l=8, m=-7, theta=p, phi=t), 
        '-6': lambda t, p: __spherical_harmonic(l=8, m=-6, theta=p, phi=t), 
        '-5': lambda t, p: __spherical_harmonic(l=8, m=-5, theta=p, phi=t), 
        '-4': lambda t, p: __spherical_harmonic(l=8, m=-4, theta=p, phi=t), 
        '-3': lambda t, p: __spherical_harmonic(l=8, m=-3, theta=p, phi=t), 
        '-2': lambda t, p: __spherical_harmonic(l=8, m=-2, theta=p, phi=t), 
        '-1': lambda t, p: __spherical_harmonic(l=8, m=-1, theta=p, phi=t), 
         '0': lambda t, p: __spherical_harmonic(l=8, m= 0, theta=p, phi=t), 
         '1': lambda t, p: __spherical_harmonic(l=8, m= 1, theta=p, phi=t), 
         '2': lambda t, p: __spherical_harmonic(l=8, m= 2, theta=p, phi=t), 
         '3': lambda t, p: __spherical_harmonic(l=8, m= 3, theta=p, phi=t), 
         '4': lambda t, p: __spherical_harmonic(l=8, m= 4, theta=p, phi=t), 
         '5': lambda t, p: __spherical_harmonic(l=8, m= 5, theta=p, phi=t), 
         '6': lambda t, p: __spherical_harmonic(l=8, m= 6, theta=p, phi=t), 
         '7': lambda t, p: __spherical_harmonic(l=8, m= 7, theta=p, phi=t), 
         '8': lambda t, p: __spherical_harmonic(l=8, m= 8, theta=p, phi=t), 
    }
}

RSH_CART = {
    '0': {
        '0': lambda x, y, z: np.sqrt(1 / 4 / np.pi) + x + y + z - x - y - z, # otherwise does not return an array 
    }, 

    '1': {
        '-1': lambda x, y, z: __spherical_harmonic(l=1, m=-1, phi=np.arccos(z), theta=np.arctan2(y,x)), # np.sqrt(3 / 4 / np.pi) * y, 
         '0': lambda x, y, z: __spherical_harmonic(l=1, m=0,  phi=np.arccos(z), theta=np.arctan2(y,x)), # np.sqrt(3 / 4 / np.pi) * z, 
         '1': lambda x, y, z: __spherical_harmonic(l=1, m=1,  phi=np.arccos(z), theta=np.arctan2(y,x)), # np.sqrt(3 / 4 / np.pi) * x, 
    }, 

    '2': {
        '-2': lambda x, y, z: __spherical_harmonic(l=2, m=-2, phi=np.arccos(z), theta=np.arctan2(y,x)), # 1 / 2 * np.sqrt(15 / np.pi) * x * y, 
        '-1': lambda x, y, z: __spherical_harmonic(l=2, m=-1, phi=np.arccos(z), theta=np.arctan2(y,x)), # 1 / 2 * np.sqrt(15 / np.pi) * y * z, 
         '0': lambda x, y, z: __spherical_harmonic(l=2, m=0,  phi=np.arccos(z), theta=np.arctan2(y,x)), # 1 / 4 * np.sqrt( 5 / np.pi) * (3 * z**2 - 1), 
         '1': lambda x, y, z: __spherical_harmonic(l=2, m=1,  phi=np.arccos(z), theta=np.arctan2(y,x)), # 1 / 2 * np.sqrt(15 / np.pi) * x * z, 
         '2': lambda x, y, z: __spherical_harmonic(l=2, m=2,  phi=np.arccos(z), theta=np.arctan2(y,x)), # 1 / 4 * np.sqrt(15 / np.pi) * (x**2 - y**2), 
    }, 

    '3': {
        '-3': lambda x, y, z: __spherical_harmonic(l=3, m=-3, phi=np.arccos(z), theta=np.arctan2(y,x)), # 1 / 4 * np.sqrt( 35 / 2 / np.pi) * y * (3 * x**2 - y**2), 
        '-2': lambda x, y, z: __spherical_harmonic(l=3, m=-2, phi=np.arccos(z), theta=np.arctan2(y,x)), # 1 / 2 * np.sqrt(105 / np.pi) * x * y * z, 
        '-1': lambda x, y, z: __spherical_harmonic(l=3, m=-1, phi=np.arccos(z), theta=np.arctan2(y,x)), # 1 / 4 * np.sqrt( 21 / 2 / np.pi) * y * (5 * z**2 - 1), 
         '0': lambda x, y, z: __spherical_harmonic(l=3, m=0,  phi=np.arccos(z), theta=np.arctan2(y,x)), # 1 / 4 * np.sqrt(  7 / np.pi) * (5 * z**3 - 3 * z), 
         '1': lambda x, y, z: __spherical_harmonic(l=3, m=1,  phi=np.arccos(z), theta=np.arctan2(y,x)), # 1 / 4 * np.sqrt( 21 / 2 / np.pi) * x * (5 * z**2 - 1), 
         '2': lambda x, y, z: __spherical_harmonic(l=3, m=2,  phi=np.arccos(z), theta=np.arctan2(y,x)), # 1 / 4 * np.sqrt(105 / np.pi) * z * (x**2 - y**2), 
         '3': lambda x, y, z: __spherical_harmonic(l=3, m=3,  phi=np.arccos(z), theta=np.arctan2(y,x)), # 1 / 4 * np.sqrt( 35 / 2 / np.pi) * x * (x**2 - 3 * y**2), 
    }, 

    '4': {
        '-4': lambda x, y, z: __spherical_harmonic(l=4, m=-4, phi=np.arccos(z), theta=np.arctan2(y,x)), # 3 / 4  * np.sqrt(35 / np.pi) * (x * y * (x**2 - y**2)), 
        '-3': lambda x, y, z: __spherical_harmonic(l=4, m=-3, phi=np.arccos(z), theta=np.arctan2(y,x)), # 3 / 4  * np.sqrt(35 / (2 * np.pi)) * y * z * (3 * x**2 - y**2), 
        '-2': lambda x, y, z: __spherical_harmonic(l=4, m=-2, phi=np.arccos(z), theta=np.arctan2(y,x)), # 3 / 4  * np.sqrt(5 / np.pi) * x * y * (7 * z**2 - 1), 
        '-1': lambda x, y, z: __spherical_harmonic(l=4, m=-1, phi=np.arccos(z), theta=np.arctan2(y,x)), # 3 / 4  * np.sqrt(5 / 2 / np.pi) * y * (7 * z**3 - 3 * z), 
         '0': lambda x, y, z: __spherical_harmonic(l=4, m=0,  phi=np.arccos(z), theta=np.arctan2(y,x)), # 3 / 16 * np.sqrt(1 / np.pi) * (35 * z**4 - 30 * z**2 + 3), 
         '1': lambda x, y, z: __spherical_harmonic(l=4, m=1,  phi=np.arccos(z), theta=np.arctan2(y,x)), # 3 / 4  * np.sqrt(5 / 2 / np.pi) * x * (7 * z**3 - 3 * z), 
         '2': lambda x, y, z: __spherical_harmonic(l=4, m=2,  phi=np.arccos(z), theta=np.arctan2(y,x)), # 3 / 8  * np.sqrt(5 / np.pi) * (7 * z**2 - 1) * (x**2 - y**2), 
         '3': lambda x, y, z: __spherical_harmonic(l=4, m=3,  phi=np.arccos(z), theta=np.arctan2(y,x)), # 3 / 4  * np.sqrt(35 / (2 * np.pi)) * x * z * (x**2 - 3 * y**2), 
         '4': lambda x, y, z: __spherical_harmonic(l=4, m=4,  phi=np.arccos(z), theta=np.arctan2(y,x)), # 3 / 16 * np.sqrt(35 / np.pi) * (x**2 * (x**2 - 3 * y**2) - y**2 * (3 * x**2 - y**2)), 
    }, 

    '5': {
        '-5': lambda x, y, z: __spherical_harmonic(l=5, m=-5, phi=np.arccos(z), theta=np.arctan2(y,x)), 
        '-4': lambda x, y, z: __spherical_harmonic(l=5, m=-4, phi=np.arccos(z), theta=np.arctan2(y,x)), 
        '-3': lambda x, y, z: __spherical_harmonic(l=5, m=-3, phi=np.arccos(z), theta=np.arctan2(y,x)), 
        '-2': lambda x, y, z: __spherical_harmonic(l=5, m=-2, phi=np.arccos(z), theta=np.arctan2(y,x)), 
        '-1': lambda x, y, z: __spherical_harmonic(l=5, m=-1, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '0': lambda x, y, z: __spherical_harmonic(l=5, m= 0, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '1': lambda x, y, z: __spherical_harmonic(l=5, m= 1, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '2': lambda x, y, z: __spherical_harmonic(l=5, m= 2, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '3': lambda x, y, z: __spherical_harmonic(l=5, m= 3, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '4': lambda x, y, z: __spherical_harmonic(l=5, m= 4, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '5': lambda x, y, z: __spherical_harmonic(l=5, m= 5, phi=np.arccos(z), theta=np.arctan2(y,x)), 
    }, 

    '6': {
        '-6': lambda x, y, z: __spherical_harmonic(l=6, m=-6, phi=np.arccos(z), theta=np.arctan2(y,x)), 
        '-5': lambda x, y, z: __spherical_harmonic(l=6, m=-5, phi=np.arccos(z), theta=np.arctan2(y,x)), 
        '-4': lambda x, y, z: __spherical_harmonic(l=6, m=-4, phi=np.arccos(z), theta=np.arctan2(y,x)), 
        '-3': lambda x, y, z: __spherical_harmonic(l=6, m=-3, phi=np.arccos(z), theta=np.arctan2(y,x)), 
        '-2': lambda x, y, z: __spherical_harmonic(l=6, m=-2, phi=np.arccos(z), theta=np.arctan2(y,x)), 
        '-1': lambda x, y, z: __spherical_harmonic(l=6, m=-1, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '0': lambda x, y, z: __spherical_harmonic(l=6, m= 0, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '1': lambda x, y, z: __spherical_harmonic(l=6, m= 1, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '2': lambda x, y, z: __spherical_harmonic(l=6, m= 2, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '3': lambda x, y, z: __spherical_harmonic(l=6, m= 3, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '4': lambda x, y, z: __spherical_harmonic(l=6, m= 4, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '5': lambda x, y, z: __spherical_harmonic(l=6, m= 5, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '6': lambda x, y, z: __spherical_harmonic(l=6, m= 6, phi=np.arccos(z), theta=np.arctan2(y,x)), 
    }, 

    '7': {
        '-7': lambda x, y, z: __spherical_harmonic(l=7, m=-7, phi=np.arccos(z), theta=np.arctan2(y,x)), 
        '-6': lambda x, y, z: __spherical_harmonic(l=7, m=-6, phi=np.arccos(z), theta=np.arctan2(y,x)), 
        '-5': lambda x, y, z: __spherical_harmonic(l=7, m=-5, phi=np.arccos(z), theta=np.arctan2(y,x)), 
        '-4': lambda x, y, z: __spherical_harmonic(l=7, m=-4, phi=np.arccos(z), theta=np.arctan2(y,x)), 
        '-3': lambda x, y, z: __spherical_harmonic(l=7, m=-3, phi=np.arccos(z), theta=np.arctan2(y,x)), 
        '-2': lambda x, y, z: __spherical_harmonic(l=7, m=-2, phi=np.arccos(z), theta=np.arctan2(y,x)), 
        '-1': lambda x, y, z: __spherical_harmonic(l=7, m=-1, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '0': lambda x, y, z: __spherical_harmonic(l=7, m= 0, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '1': lambda x, y, z: __spherical_harmonic(l=7, m= 1, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '2': lambda x, y, z: __spherical_harmonic(l=7, m= 2, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '3': lambda x, y, z: __spherical_harmonic(l=7, m= 3, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '4': lambda x, y, z: __spherical_harmonic(l=7, m= 4, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '5': lambda x, y, z: __spherical_harmonic(l=7, m= 5, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '6': lambda x, y, z: __spherical_harmonic(l=7, m= 6, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '7': lambda x, y, z: __spherical_harmonic(l=7, m= 7, phi=np.arccos(z), theta=np.arctan2(y,x)), 
    }, 

    '8': {
        '-8': lambda x, y, z: __spherical_harmonic(l=8, m=-8, phi=np.arccos(z), theta=np.arctan2(y,x)), 
        '-7': lambda x, y, z: __spherical_harmonic(l=8, m=-7, phi=np.arccos(z), theta=np.arctan2(y,x)), 
        '-6': lambda x, y, z: __spherical_harmonic(l=8, m=-6, phi=np.arccos(z), theta=np.arctan2(y,x)), 
        '-5': lambda x, y, z: __spherical_harmonic(l=8, m=-5, phi=np.arccos(z), theta=np.arctan2(y,x)), 
        '-4': lambda x, y, z: __spherical_harmonic(l=8, m=-4, phi=np.arccos(z), theta=np.arctan2(y,x)), 
        '-3': lambda x, y, z: __spherical_harmonic(l=8, m=-3, phi=np.arccos(z), theta=np.arctan2(y,x)), 
        '-2': lambda x, y, z: __spherical_harmonic(l=8, m=-2, phi=np.arccos(z), theta=np.arctan2(y,x)), 
        '-1': lambda x, y, z: __spherical_harmonic(l=8, m=-1, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '0': lambda x, y, z: __spherical_harmonic(l=8, m= 0, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '1': lambda x, y, z: __spherical_harmonic(l=8, m= 1, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '2': lambda x, y, z: __spherical_harmonic(l=8, m= 2, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '3': lambda x, y, z: __spherical_harmonic(l=8, m= 3, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '4': lambda x, y, z: __spherical_harmonic(l=8, m= 4, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '5': lambda x, y, z: __spherical_harmonic(l=8, m= 5, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '6': lambda x, y, z: __spherical_harmonic(l=8, m= 6, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '7': lambda x, y, z: __spherical_harmonic(l=8, m= 7, phi=np.arccos(z), theta=np.arctan2(y,x)), 
         '8': lambda x, y, z: __spherical_harmonic(l=8, m= 8, phi=np.arccos(z), theta=np.arctan2(y,x)), 
    }
}


Y00  = lambda x, y, z: np.sqrt(1 / 4 / np.pi)

Y10  = lambda x, y, z: np.sqrt(3 / 4 / np.pi) * z
Y1m1 = lambda x, y, z: np.sqrt(3 / 4 / np.pi) * y
Y11  = lambda x, y, z: np.sqrt(3 / 4 / np.pi) * x

Y20  = lambda x, y, z: 1 / 4 * np.sqrt(5 / np.pi) * (3 * z**2 - 1)
Y2m1 = lambda x, y, z: 1 / 2 * np.sqrt(15 / np.pi) * y * z
Y21  = lambda x, y, z: 1 / 2 * np.sqrt(15 / np.pi) * x * z
Y2m2 = lambda x, y, z: 1 / 2 * np.sqrt(15 / np.pi) * x * y
Y22  = lambda x, y, z: 1 / 4 * np.sqrt(15 / np.pi) * (x**2 - y**2)

Y30  = lambda x, y, z: 1 / 4 * np.sqrt(7 / np.pi) * (5 * z**3 - 3 * z)
Y3m1 = lambda x, y, z: 1 / 4 * np.sqrt(21 / (2 * np.pi)) * y * (5 * z**2 - 1)
Y31  = lambda x, y, z: 1 / 4 * np.sqrt(21 / (2 * np.pi)) * x * (5 * z**2 - 1)
Y3m2 = lambda x, y, z: 1 / 2 * np.sqrt(105 / np.pi) * x * y * z
Y32  = lambda x, y, z: 1 / 4 * np.sqrt(105 / np.pi) * z * (x**2 - y**2)
Y3m3 = lambda x, y, z: 1 / 4 * np.sqrt(35 / (2 * np.pi)) * y * (3 * x**2 - y**2)
Y33  = lambda x, y, z: 1 / 4 * np.sqrt(35 / (2 * np.pi)) * x * (x**2 - 3 * y**2)

Y40  = lambda x, y, z: 3 / 16 * np.sqrt(1 / np.pi) * (35 * z**4 - 30 * z**2 + 3)
Y4m1 = lambda x, y, z: 3 / 4  * np.sqrt(5 / 2 / np.pi) * y * (7 * z**3 - 3 * z)
Y41  = lambda x, y, z: 3 / 4  * np.sqrt(5 / 2 / np.pi) * x * (7 * z**3 - 3 * z)
Y4m2 = lambda x, y, z: 3 / 4  * np.sqrt(5 / np.pi) * x * y * (7 * z**2 - 1)
Y42  = lambda x, y, z: 3 / 8  * np.sqrt(5 / np.pi) * (7 * z**2 - 1) * (x**2 - y**2)
Y4m3 = lambda x, y, z: 3 / 4  * np.sqrt(35 / (2 * np.pi)) * y * z * (3 * x**2 - y**2)
Y43  = lambda x, y, z: 3 / 4  * np.sqrt(35 / (2 * np.pi)) * x * z * (x**2 - 3 * y**2)
Y4m4 = lambda x, y, z: 3 / 4  * np.sqrt(35 / np.pi) * (x * y * (x**2 - y**2))
Y44  = lambda x, y, z: 3 / 16 * np.sqrt(35 / np.pi) * (x**2 * (x**2 - 3 * y**2) - y**2 * (3 * x**2 - y**2))


lm = [(0, 0), 
      (1,-1), (1, 0), (1, 1),
      (2,-2), (2,-1), (2, 0), (2, 1), (2, 2), 
      (3,-3), (3,-2), (3,-1), (3, 0), (3, 1), (3, 2), (3, 3), 
      (4,-4), (4,-3), (4,-2), (4,-1), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), 
      (5,-5), (5,-4), (5,-3), (5,-2), (5,-1), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), 
      (6,-6), (6,-5), (6,-4), (6,-3), (6,-2), (6,-1), (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), 
      (7,-7), (7,-6), (7,-5), (7,-4), (7,-3), (7,-2), (7,-1), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), 
      (8,-8), (8,-7), (8,-6), (8,-5), (8,-4), (8,-3), (8,-2), (8,-1), (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), ]


def test_check_orthonormality():
    # 进行正交归一性检查
    from scipy.integrate import lebedev_rule
    global lm
    xleb, wleb = lebedev_rule(35) # 434
    print('>>> 检查实球谐函数正交归一性...')
    for (l, m) in lm:
        print(f'    l = {l:>2d}', end='\r')
        for (ll, mm) in lm:
            integral_sph = np.einsum('i,i,i->', RSH_SPH[f'{l}'][f'{m}'](t=np.arccos(xleb[2]), p=np.arctan2(xleb[1], xleb[0])), 
                                     RSH_SPH[f'{ll}'][f'{mm}'](t=np.arccos(xleb[2]), p=np.arctan2(xleb[1], xleb[0])), 
                                     wleb)
            integral = np.einsum('i,i,i->', RSH_CART[f'{l}'][f'{m}'](*xleb), RSH_CART[f'{ll}'][f'{mm}'](*xleb), wleb)
            # integral = np.einsum('i,i,i->', 
            #                      __spherical_harmonic_cart(l, m, xleb), 
            #                      __spherical_harmonic_cart(ll, mm, xleb), wleb)
            if (l == ll) and (m == mm):
                if not np.isclose(integral, 1., atol=1e-12):
                    print((l,m), (ll,mm), integral)
                    raise Exception
                if not np.isclose(integral_sph, 1., atol=1e-12):
                    print((l,m), (ll,mm), integral_sph)
                    raise Exception
            else:
                if not np.isclose(integral, 0., atol=1e-12):
                    print((l,m), (ll,mm), integral)
                    raise Exception
                if not np.isclose(integral_sph, 0., atol=1e-12):
                    print((l,m), (ll,mm), integral_sph)
                    raise Exception
    print('\n>>> 检查完毕\n')
                    
def test_rsh_return_realval():
    # 检查实球谐函数是否返回实数
    from scipy.integrate import lebedev_rule
    global lm
    xleb, wleb = lebedev_rule(35) # 434
    l = 8
    vals = [RSH_CART[f'{l}'][f'{m}'](*xleb) for m in range(-l,l+1)]
    for val in vals:
        for vv in val:
            if not isinstance(vv, float):
                print(vv)

def test():
    from llebedev import lebedev_rule
    xleb, wleb = lebedev_rule(17)
    thetas = np.arccos(xleb[2])
    phis = np.arctan2(xleb[1], xleb[0])
    res = rsh(3, 1, thetas, phis)
    for theta, phi, r in zip(thetas, phis, res):
        print(f"({theta:.15f}, {phi:.15f})    {r:.15f}")

def test2():
    l = 14
    m = -11
    thetas = np.linspace(1., 10., 10)
    phis = np.linspace(0.1, 1., 10)
    for theta, phi in zip(thetas, phis):
        print(f"{rsh(l, m, theta, phi):<20.15f}")


if __name__ == '__main__':
    # test_check_orthonormality()
    # test_rsh_return_realval()
    test2()
