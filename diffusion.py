# diffusion.py
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import linalg, sparse
from scipy.sparse import linalg

# Handy function
def pause(message=''):
    print(message)
    programPause = raw_input("Press <ENTER> to continue...")

def euler(y,a,T,Bcondition='dirichlet_zero',scheme='implicit'):
    '''Euler scheme with Au(n+1)=Bu(n).'''
    if scheme=='explicit':
        B = sparse.diags([a, 1-2*a, a], [-1, 0, 1], shape=(len(y), len(y)), format='csr')
        if Bcondition == 'dirichlet_zero':
            # Completely absorbing boundary conditions.
            A = sparse.eye(len(y-2),format='csr')
        elif Bcondition == 'neumann_zero':
            A = sparse.eye(len(y),format='csr')
            B[0,1]=B[-1,-2]=2*a
        else:
            print('Invalid boundary conditions.')
    elif scheme=='implicit':
        A = sparse.diags([-a, 1+2*a, -a], [-1, 0, 1], shape=(len(y), len(y)), format='csr')
        if Bcondition == 'dirichlet_zero':
            # Completely absorbing boundary conditions.
            B = sparse.eye(len(y-2),format='csr')
        elif Bcondition=='neumann_zero':
            # Completely reflecting boundary conditions.
            A[0,1]=A[-1,-2]=-2*a
            B = sparse.eye(len(y),format='csr')
        else:
            print('Invalid boundary conditions.')
    elif scheme=='crank-nicolson':
        A = sparse.diags([-a/2, 1+a, -a/2], [-1, 0, 1], shape=(len(y), len(y)), format='csr')
        B = sparse.diags([a/2, 1-a, a/2], [-1, 0, 1], shape=(len(y), len(y)), format='csr')
        # Note, if completely absorbing boundary conditions (dirichlet_zero), nothing more to do with A,B.
        if Bcondition=='neumann_zero':
            # Completely reflecting boundary conditions.
            A[0,1]=A[-1,-2]=-a
            B[0,1]=B[-1,-2]=a
        elif not Bcondition=='dirichlet_zero':
            print('Invalid boundary conditions.')
    else:
        print('Invalid scheme: ', scheme)

    for t in range(1,T):
        y = scipy.sparse.linalg.spsolve(A,B.dot(y))
    return y

def advection_LaxW(y,a,T):
    b = 2.*a**2.
    B = sparse.diags([a+b, 1-2*b, -a+b], [-1,0,1], shape=(len(y), len(y)), format='csr')
    for t in range(1,T):
        y = B.dot(y)
    return y

if __name__ == "__main__":

    # Testing advection and Hopf equations
    c = 1
    delta_t = 0.01
    n_steps = 10
    CLF = 0.8
    a = CLF/2.
    delta_x = c*delta_t/(2*a)
    x_steps = 1/delta_x

    x = np.linspace(0,1,x_steps)
    y = np.zeros([x_steps])

    for i in range(1,len(y)):
        y[i] = max( 1- (4.*(x[i]-0.5) )**2. , 0)

    plt.plot(x,y)
    y = advection_LaxW(y,a,n_steps)
    plt.plot(x,y)
    y = advection_LaxW(y,a,n_steps)
    plt.plot(x,y)

    # Testing Euler scheme
    # n_points  = 100
    # n_steps = 10
    # a = 2.5
    #
    # Bcondition = 'dirichlet_zero'
    # # Bcondition = 'neumann_zero'
    # # scheme = 'explicit'
    # # scheme = 'implicit'
    # scheme = 'crank-nicolson'
    #
    # x = np.linspace(0,1,n_points+2)
    # y = np.zeros([n_points + 2])
    #
    # y[round(len(y)/2)] = 1./n_points
    # # for i in range(0,len(y)):
    # #     y[i]=i
    # # plt.plot(x,y)
    #
    # y = euler(y,a,n_steps,Bcondition=Bcondition,scheme=scheme)
    # plt.plot(x,y)
    # y = euler(y,a,n_steps,Bcondition=Bcondition,scheme=scheme)
    # plt.plot(x,y)
    # y = euler(y,a,n_steps*10,Bcondition=Bcondition,scheme=scheme)
    # plt.plot(x,y)

    plt.show(block=False)
    pause('Showing function.')
