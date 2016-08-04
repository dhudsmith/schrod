# schrod
### *A fast, accurate, and simple module for solving the single particle Schrodinger equations in 1, 2, and 3 dimensions.*

# Features:

1. Simple interface:
```python
import schrodinger
import numpy as np

# Specify the potential
x = np.linspace(-10, 10, 200)
V = 1/2. * x**2

# Create and solve Schrodinger's equation
eqn = schrod.Schrod(x, V)
eqn.solve()

# Get the results
print(sch_eqn.eigs[0:3])
>> [  0.50008379   1.50064933   2.51314075  ]
```
