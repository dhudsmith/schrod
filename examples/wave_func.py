"""
Solve Schodinger's equation and plot the eigenfunctions
Author: D. Hudson Smith
"""

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import schrod

# Specify the potential
x = np.linspace(-3, 3, 200)
V = x + 0.5*x**4

# Create and solve Schrodinger's equation
eqn = schrod.Schrod(x, V, n_basis=40)
eqn.solve()

# Get the eigenvalues and eigenvectors
eig_vals = eqn.eigs
psis = eqn.psi_eig_x()

# Plot everything
plt.xlim((x[0],x[-1]))
plt.ylim((-1,9))
plt.title("Wave functions for\n$V(x) = x + 0.5 x^4$", fontsize=18)
plt.xlabel("$x$")
plt.ylabel('Energy')

n_eigs_plt = 5
plt.plot(x, V, 'b-', lw=2)
for i in range(n_eigs_plt):
    plt.plot(x, eig_vals[i]+0.75*psis[i], color='red', lw=1.5)
    plt.axhline(eig_vals[i], ls='--', color='black', lw=2)

# Optionally show/save
# plt.show()
plt.tight_layout()
plt.savefig("./plots/wave_func.png", dpi=100)