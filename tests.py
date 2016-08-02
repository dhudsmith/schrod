import tise
import time
import numpy as np
import matplotlib.pyplot as plt

def timeit(x, V, n_basis_vec):

    # Initialize the tise
    se = tise.tise(x, V)

    # Time the tise for different numbers of basis states
    times=[]
    for n in n_basis_vec:
        se.set_n_basis(n)
        ti = time.time()
        se.eigsys()
        tf = time.time()
        times.append(tf-ti)

    return times

if __name__=="__main__":

    # The harmonic oscillator
    x_vec = np.linspace(-1, 1, 200)
    V_vec = np.asarray(([1000 * x ** 2 for x in x_vec],))

    # Time it:
    ns = np.arange(5,206,10)
    times = timeit(x_vec, V_vec, ns)
    plt.loglog(ns, times, 'bo')
    plt.show()
    plt.clf()
