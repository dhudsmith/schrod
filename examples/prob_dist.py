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
probs = eqn.prob_eig_x()

# Plot everything
plt.xlim((x[0],x[-1]))
plt.ylim((-1,9))
plt.title("Probability distributions for\n$V(x) = x + 0.5 x^4$", fontsize=18)
plt.xlabel("$x$")
plt.ylabel('Energy')

n_eigs_plt = 5
plt.plot(x, V, 'b-', lw=2)
for i in range(n_eigs_plt):
    plt.fill_between(x, eig_vals[i], eig_vals[i]+probs[i], color='red', lw=0)
    plt.axhline(eig_vals[i], ls='--', color='black', lw=2)

# Optionally show/save
# plt.show()
plt.tight_layout()
plt.savefig("./plots/prob_dist.png", dpi=100)