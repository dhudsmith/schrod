# schrod
### *A simple, accurate, and fast module for solving the single particle Schrodinger equation.*
<img src="/examples/plots/prob_dist.png" alt="Probability distributions for V(x) = x + 0.5*x^4" width="500"/>

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
The harmonic oscillator:
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

For more examples, see the [examples](examples/) folder.

~~~~ Testing trigger event yet again ~~~~
