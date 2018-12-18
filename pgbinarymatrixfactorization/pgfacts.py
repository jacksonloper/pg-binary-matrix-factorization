import numpy as np


def pge_safe(c):
    '''
    let Y~ PG(1,c)

    We return 

            E[Y]

    Abstractly, this is given by 

        np.tanh(c/2)/(2*c)

    But that's not always perfectly numerically stable.

    '''

    if isinstance(c,np.ndarray):
        rez=np.zeros(c.shape)
        good=(np.abs(c)>1e-9)

        goodc=c[good]
        rez[good]=np.tanh(goodc/2)/(2*goodc)
        rez[~good]=.25
        return rez
    else:
        return np.tanh(c/2)/(2*c)

def pkl(c):
    '''
    We return

    KL(PG(1,c) || PG(1,0))

    '''

    return np.log(np.cosh(c/2))-np.tanh(c/2)*c/4