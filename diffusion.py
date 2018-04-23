# diffusion.py
import numpy as np
import matplotlib.pyplot as plt
import copy

# Handy function
def pause(message=''):
    print(message)
    programPause = raw_input("Press <ENTER> to continue...")


def euler_explicit(y,a,T,condition='dirichlet_zero'):
    ''' Solves dy = a*dy/dx with explicit Euler method.
    T: number of time steps to take.
    y: initial state. '''
    if condition == 'dirichlet_zero':
        # Completely absorbing boundary condition.
        # Check that vector is correctly initiated
        if y[0]!=0 or y[-1]!=0:
            print("Absorbing Dirichlet BCs need y[0]=y[-1]=0, y was changed accordingly.")
            y[0]=y[-1]=0
        for t in range(1,T):
            u = copy.copy(y) # Inefficient, but need to not get shift of result
            for i in range(1,len(y)-1):
                y[i] = u[i] + a * (u[i+1] - 2*u[i] + u[i-1])
    return y





if __name__ == "__main__":
    n_points  = 500
    n_steps = 10
    a = 0.27

    x = np.linspace(0,1,n_points+2)
    y = np.zeros([n_points + 2])

    y[round(len(y)/2)] = 1./n_points
    # for i in range(0,102):
    #     y[i]=i
    plt.plot(x,y)

    euler_explicit(y,a,n_steps,condition='dirichlet_zero')
    plt.plot(x,y)
    euler_explicit(y,a,n_steps,condition='dirichlet_zero')
    plt.plot(x,y)
    euler_explicit(y,a,n_steps*20,condition='dirichlet_zero')
    plt.plot(x,y)

    plt.show(block=False)
    pause('Showing function.')
