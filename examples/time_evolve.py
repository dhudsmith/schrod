import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import schrod

######################################################################
# Set and solve Schrodinger's Equation

# Specify the potential
x = np.linspace(-10, 10, 400)
# V = 1/2 * x**2
V = np.zeros(400)

# Create and solve Schrodinger's equation
eqn = schrod.Schrod(x, V, n_basis=75)
sol = eqn.solve()

# Print the first five eigenvalues
print("Eigenvalues: ", eqn.eigs[0:5])

######################################################################
# The initial wave packet


# Helper functions for gaussian wave-packets
def gauss_x(x, a, x0, k0):
    """
    a gaussian wave packet of width a, centered at x0, with momentum k0
    """
    return ((a * np.sqrt(np.pi)) ** (-0.5)
            * np.exp(-0.5 * ((x - x0) * 1. / a) ** 2 + 1j * x * k0))

# The wave packet parameters
VL = 1 / (x[0] - x[-1]) ** 2
k0 = 40 * np.sqrt(VL)
sigma = 0.1 * (x[-1] - x[0])
x0 = 0

# The actual wavepacket
psi_0 = gauss_x(x, sigma, x0, k0)

######################################################################
# Time evolve

ts = np.linspace(0, 5 * k0 ** 2, 300)
prob_tx = eqn.prob_tx(psi_0, ts)
prob_tk = eqn.prob_tk(psi_0, ts)

######################################################################
# Animate

fig = plt.figure()

# Coordinate space
ax1 = fig.add_subplot(211, autoscale_on=False, xlim=(x[0], x[-1]), ylim=(0, 1))
line_prob_xt, = ax1.plot([], [], 'r-', lw=2)
line_Vx, = ax1.plot([], [], 'b-', lw=2)
line_Vx.set_data(eqn.x, eqn.V)
xdata, ydata = [], []

# Momentum space
ymin = 0
ymax = prob_tk.max()
ax2 = fig.add_subplot(212, xlim = (eqn.k[0], eqn.k[-1]), ylim=(ymin, ymax))
line_prob_kt, =  ax2.plot([], [], 'r-', lw=2)


def init():
    line_prob_xt.set_data([], [])
    line_Vx.set_data([], [])
    line_prob_kt.set_data([], [])
    return line_prob_xt, line_Vx, line_prob_kt


def update_line(i):
    line_prob_xt.set_data(x, prob_tx[i])
    line_prob_kt.set_data(eqn.k, prob_tk[i])
    line_Vx.set_data(x, V)
    return line_prob_xt, line_prob_kt, line_Vx


ani = animation.FuncAnimation(fig, update_line,
                              np.arange(0, prob_tx.shape[0]),
                              interval=15,
                              init_func=init,
                              blit=True)

ani.save('animations/time_evolve.mp4', fps=20)
