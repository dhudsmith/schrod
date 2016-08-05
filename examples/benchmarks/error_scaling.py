import matplotlib.pyplot as plt
import numpy as np
import schrod

# A simple error calculation function
def error(x, V, n_basis_vec, exact):

    # Initialize the Schrod
    eqn = schrod.Schrod(x, V)

    # Calculate the error
    rel_err=[]
    for n in n_basis_vec:
        eqn.set_n_basis(n)
        eqn.solve()
        lam = eqn.eigs
        err = (np.abs((lam - exact[0:n])/exact[0:n]))[...,0:n_basis_vec[0]]

        rel_err.append(err)

    return np.asarray(rel_err)

if __name__=="__main__":

    # Specify the potential
    x = np.linspace(-5, 5, 200)
    V = 1 / 2 * x ** 2

    # The basis states
    ns = np.logspace(np.log10(4), np.log10(100), 15, dtype=np.int16)
    print("Tests will be run for the following numbers "
          "of basis states:\n", ns)

    # Calculate the error
    exact_energies = np.arange(0, np.max(ns)) + 1/2
    err = error(x, V, ns, exact_energies)

    # Plot options
    plt.xlabel("$\log_{10}(n_{basis})$")
    plt.ylabel("$\log_{10}(err)$")
    plt.title("Numeric and systematic errors\n for the smallest 4 eigenvalues.")
    plt.ylim((-10,0.2))

    # Plot the error for each eigenvalue
    for i in range(ns[0]):
        # The points
        plt.plot(np.log10(ns), np.log10(err[:,i]), 'o-',
                 label="Measured errors")

    # Annotations
    plt.axvline(1.3, ls=':', lw=2, color='k')

    ax = plt.axes()
    bbox_props = dict(boxstyle="square,pad=0.1", fc="w", ec="k", lw=1)
    arrow_props = dict(facecolor='black', shrink=0.05)
    ax.annotate('Numerical error\nregime',
                xy=(1.1, -4), xytext=(1.2, -2),
                arrowprops=arrow_props,
                bbox=bbox_props,
                fontsize=20
                )
    ax.annotate('Hard wall system-\natic error regime',
                xy=(1.4, -6), xytext=(1.4, -4),
                arrowprops=arrow_props,
                bbox=bbox_props,
                fontsize=20
                )

    # Optionally show/save
    # plt.show()
    plt.savefig("./plots/error_scaling.png")