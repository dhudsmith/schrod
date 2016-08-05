# schrod
### *A simple, accurate, and fast module for solving the single particle Schrodinger equation.*

## Installation

1. Make sure you have `python3`, `numpy`, and `scipy`. All of the these (and many more) are included with [anaconda](https://www.continuum.io/downloads).
2. Clone the repo:

 ```bash
 git clone https://github.com/dhudsmith/schrod.git
 ```
3. From inside the `schrod` directory execute

 ```
 python setup.py install --user
 ```
Optionally you can omit `--user` to install for all users.

That's it. To load the package into a python shell, simply execute
```python
import schrod
```

## Usage
### The harmonic oscillator
```python
import schrod, numpy

# Specify the potential
x = numpy.linspace(-5, 5, 200)
V = 1/2 * x**2

# Create and solve Schrodinger's equation
eqn = schrod.Schrod(x, V)
eqn.solve()

# Print the first five eigenvalues
print(eqn.eigs[0:5])
```
Output:
`[ 0.5         1.5         2.50000008  3.50000122  4.50001267]`

###Specify the desired error tolerance
```python
import schrod, numpy

# Specify the potential
x = numpy.linspace(-5, 5, 200)
V = 3*numpy.sin(5*x)

# Create and solve Schrodinger's equation
eqn = schrod.Schrod(x, V)
sol = eqn.solve_to_tol(n_eig=5, tol=1e-10, n_init=100, n_max=200)

# Print the first five eigenvalues
print("Eigenvalues: \n", eqn.eigs[0:5])
print("Relative error estimate: \n", sol.eig_errs)
print("Number of basis states at convergence: \n", sol.n_basis_converged)
```
Output:
```
Eigenvalues: 
 [-0.30730421 -0.17601668  0.03925543  0.3314096   0.68529923]
Error estimate: 
 [  8.15585461e-13   9.92877417e-12   5.43644143e-11   1.67203654e-11
   7.07640390e-12]
Number of basis states at convergence: 
 145
```
