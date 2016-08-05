import matplotlib.pyplot as plt
import numpy as np
import schrod
import time

# A simple timing function:
def timing(x, V, n_basis_vec, n_tests=1):

    # Initialize the Schrod
    eqn = schrod.Schrod(x, V)

    # Time the Schrod for different numbers of basis states
    times=[]
    for n in n_basis_vec:
        avg_time = 0
        for i in range(n_tests):
            eqn.set_n_basis(n)
            ti = time.time()
            eqn.solve()
            tf = time.time()
            avg_time += (tf - ti)/n_tests

        times.append(avg_time)

    return times

# Specify the potential
x = np.linspace(-5, 5, 200)
V = 1 / 2 * x ** 2

# The number of basis states to scan over
ns = np.logspace(np.log10(4), np.log10(100), 15, dtype=np.int16)
print("Tests will be run for the following numbers "
      "of basis states:\n",
      ns)

# Calculate how long it takes for each choice of number of basis states
times = timing(x, V, ns, n_tests=10)

# Plot the points and a fit line
plt.plot(np.log10(ns), np.log10(times), 'bo',
         label="Measured times")
m, b = np.polyfit(np.log10(ns), np.log10(times), 1)
plt.plot(np.log10(ns), m * np.log10(ns) + b, 'r-', lw=2,
         label = "%0.3f $\log_{10}(n_{basis})$ + %0.3f" % (m,b))

# Plot options
plt.legend(loc=2)
plt.xlabel("$\log_{10}(n_{basis})$")
plt.ylabel("$\log_{10}(seconds)$")
plt.title("Quadratic time scaling")

# Optionally show/save
#plt.show()
plt.savefig("./plots/time_scaling.png")
