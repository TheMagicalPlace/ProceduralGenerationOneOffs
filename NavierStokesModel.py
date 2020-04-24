from random import randint
from collections import defaultdict
from typing import List,TypeVar,Tuple,Dict

import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt

nlist = np.array
nrand = np.linspace
pi = np.pi

class NavierStokesModel:


    def __init__(self):
        pass

if __name__ == '__main__':
    zend = 100

    Z = np.linspace(1,zend,25,True)
    vmean = 1
    vmean0 =10
    vmeanlast = 1
    a = 1
    viscosity = .050
    fric_coeff = 0.5
    density = 1

    P = lambda z: 1/(2*a)*fric_coeff*density*vmean**2/(2*9.81)
    uz = lambda r,z : pi*a**2/(4*viscosity)*(1-r**2/a**2)*P(z)
    rr = np.linspace(-a,a)
    normconst = max(uz(rr,1))
    plt.figure()
    d = []
    for i,z in enumerate(Z):
        d.append(P(z))
        res = uz(rr,z)
        vmean = vmeanlast - np.mean(res)
        norm = res/normconst*(zend/len(Z))
        ynorm = rr/max(rr)*.95
        plt.plot([z for _ in res],ynorm)
        plt.plot(z+norm,ynorm)
        vmeanlast = vmean

    plt.show()
