from __future__ import print_function

import schrod, numpy

# Specify the potential
x = numpy.linspace(-5, 5, 200)
V = 1/2 * x**2

# Create and solve Schrodinger's equation
eqn = schrod.Schrod(x, V)
eqn.solve()

# Print the first five eigenvalues
print(eqn.eigs[0:5])