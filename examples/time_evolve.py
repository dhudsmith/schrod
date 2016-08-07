import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import schrod

# Specify the potential
x = np.linspace(-10, 10, 300)
# V = 1/2 * x**2
V = np.zeros(300)

# Create and solve Schrodinger's equation
eqn = schrod.Schrod(x, V, n_basis=75)
sol = eqn.solve()

# Print the first five eigenvalues
print("Eigenvalues: ", eqn.eigs[0:5])

# The initial wave packet
# Helper functions for gaussian wave-packets
def gauss_x(x, a, x0, k0):
    """
    a gaussian wave packet of width a, centered at x0, with momentum k0
    """
    return ((a * np.sqrt(np.pi)) ** (-0.5)
            * np.exp(-0.5 * ((x - x0) * 1. / a) ** 2 + 1j * x * k0))

# The wave packet parameters
VL = 1/(x[0]-x[-1])**2
k0 = 50 * np.sqrt(VL)
sigma = 0.1 * (x[-1]-x[0])
x0=0

# The actual wavepacket
psi_0 = gauss_x(x, sigma, x0, k0)

# Time evolve
ts = np.linspace(0, 5*k0**2, 300)
psi_of_t = eqn.prob_tx(psi_0, ts)

# Animate
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(x[0], x[-1]), ylim=(0, 1))
ax.plot(x, V, 'b-')

line, = ax.plot([],[], 'r-', lw=2)
xdata, ydata = [], []

def init():
    line.set_data([], [])
    return line,

def update_line(i):
    thisy = psi_of_t[i]
    line.set_data(x, thisy)
    return line,

ani = animation.FuncAnimation(fig, update_line,
                              np.arange(0, psi_of_t.shape[0]),
                              interval=15,
                              init_func=init,
                              blit=True)

ani.save('animations/time_evolve.mp4', fps=20)