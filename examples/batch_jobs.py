# We can efficiently solve for multiple potentials with one command
# Under the hood, using the batch functionality is more efficient
# than solving for one potential at a time because executing in batch
# doesn't require transitioning back in forth between the python
# and the C backend between jobs.
from __future__ import print_function

import schrod
import numpy as np

# Specify the 10 ho potentials
x = np.linspace(-5, 5, 200)
ws=np.arange(1,11)
V = 1/2 * np.outer(ws**2, x**2)

# Create and solve Schrodinger's equation
eqn = schrod.Schrod(x, V, n_basis=40)
eqn.solve()

# Print the first five eigenvalues
print(eqn.eigs[...,0:5])
