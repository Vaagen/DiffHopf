# diffusion.py
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import linalg, sparse
from scipy.sparse import linalg, csr_matrix

# Handy function
def pause(message=''):
    print(message)
    programPause = raw_input("Press <ENTER> to continue...")

def euler(y,a,T,condition='dirichlet_zero',scheme='implicit'):
    '''Euler scheme with Au(n+1)=Bu(n).'''
    if scheme=='implicit':
        if condition == 'dirichlet_zero':
            # Completely absorbing boundary condition.
            A = scipy.sparse.eye(len(y-2),format='csc')
            B = sparse.diags([a, 1-2*a, a], [-1, 0, 1], shape=(len(y), len(y)), format='csc').toarray()
    Ainv = sparse.linalg.inv(A)
    for t in range(1,T):
        y = Ainv.dot(B.dot(y))
    return y



if __name__ == "__main__":

    n_points  = 100
    n_steps = 100
    a = 0.27

    x = np.linspace(0,1,n_points+2)
    y = np.zeros([n_points + 2])

    y[round(len(y)/2)] = 1./n_points
    for i in range(0,len(y)):
        y[i]=i
    plt.plot(x,y)

    y = euler(y,a,n_steps,condition='dirichlet_zero',scheme='implicit')
    plt.plot(x,y)
    y = euler(y,a,n_steps,condition='dirichlet_zero',scheme='implicit')
    plt.plot(x,y)
    y = euler(y,a,n_steps*10,condition='dirichlet_zero',scheme='implicit')
    plt.plot(x,y)

    plt.show(block=False)
    pause('Showing function.')
