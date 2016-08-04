# schrod
### *A simple, accurate, and fast module for solving the single particle Schrodinger equations in 1, 2, and 3 dimensions.*

# Features:

1. Simple interface:
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

2. Accurate:

3. Fast:
