from __future__ import print_function

import schrod, numpy

# Specify the potential
x = numpy.linspace(-20, 20, 200)
V = 1/2 * x**2

# Ask for 10 digits of precision
eqn = schrod.Schrod(x, V)
sol = eqn.solve_to_tol(n_eig=5, tol=1e-10, n_init=50, n_max=150)

# Print the first five eigenvalues
numpy.set_printoptions(15)
print("Eigenvalues: \n", eqn.eigs[0:5],
      "\nError estimate: \n", sol.eig_errs,
      "\nNumber of basis states at convergence: \n", sol.n_basis_converged)

# If <tol> cannot be achieved, a warning is returned:
sol = eqn.solve_to_tol(n_eig=5, tol=1e-15, n_init=100, n_max=150, n_step=10)

# Print the first five eigenvalues
print("Eigenvalues: \n", eqn.eigs[0:5],
      "\nError estimate: \n", sol.eig_errs,
      "\nNumber of basis states at termination: \n", sol.n_basis_converged)

# Sample output
# Eigenvalues:
#  [ 0.499999999999998  1.500000000000019  2.499999999999991
#   3.500000000000062  4.500000000000114]
# Error estimate:
#  [  1.332267629550185e-14   1.687538997430230e-14   3.197442310920411e-14
#    1.960336654909335e-13   7.033188846107148e-12]
# Number of basis states at convergence:
#  85
#
# Eigenvalues:
#  [ 0.499999999999989  1.499999999999903  2.500000000000028
#   3.499999999999986  4.500000000000009]
# Error estimate:
#  [  3.297362383136731e-14   5.684341886081010e-14   1.705302565824236e-14
#    2.410769996328924e-15   9.868649107779154e-16]
# Number of basis states at termination:
#  150
# UserWarning: Unable to achieve desired tolerance of 1.00e-15.
# Achieved a tolerance of 5.68e-14.
# Try increasing the maximum number of basis states.
#   "Try increasing the maximum number of basis states." % (tol, measured_tol))