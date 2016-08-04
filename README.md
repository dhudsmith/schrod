# Schrodinger
### *A fast, accurate, and simple module for solving the single particle Schrodinger equations in 1, 2, and 3 dimensions.*

# Features:

1. Easy to use interface:
```python
$ x = np.linspace{-2,2,100}
$ V = 1/2. * x**2
$ schrod = Schrodinger.oned(x, V)
$ schrod.solve()
$ print(solution.eig_vals)
$ >> 
```
