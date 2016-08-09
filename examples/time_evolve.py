"""
Solve and animate the Schrodinger equation.
Author: D. Hudson Smith

Acknowledgements:
Animation code relies heavily upon https://github.com/jakevdp/pySchrodinger
First presented at http://jakevdp.github.com/blog/2012/09/05/quantum-python/
Author: Jake Vanderplas <vanderplas@astro.washington.edu>
License: BSD
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import schrod

# Note: this example is under construction!

######################################################################
# Set and solve Schrodinger's Equation

# Specify the potential
def Vfunc(x):
    return np.abs(x)

x = np.linspace(-30, 30, 600)
V = Vfunc(x)
Vmax = V[0]/3

# Create and solve Schrodinger's equation
eqn = schrod.Schrod(x, V, n_basis=100)
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
k0 = np.sqrt(Vmax/3)
sigma = 0.05 * eqn.box_size
x0 = 0

# The actual wavepacket
psi_0 = gauss_x(x, sigma, x0, k0)

E = eqn.expected_E(psi_0)

######################################################################
# Time evolve
tclass = eqn.box_size/k0
ts = np.linspace(0, 0.75*tclass, 600)
prob_tx = eqn.prob_tx(psi_0, ts)
prob_tk = eqn.prob_tk(psi_0, ts)

######################################################################
# Animate

fig = plt.figure()

# Coordinate space
xmin = x[0]/3
xmax = x[-1]/3
ymin = 0
ymax = Vmax
ax1 = fig.add_subplot(211,
                      autoscale_on=False,
                      xlim=(xmin, xmax), ylim=(ymin, ymax))
line_prob_xt, = ax1.plot([], [], 'r-', lw=2, label='$|\psi(x,t)|^2$')
line_Vx, = ax1.plot([], [], 'b-', lw=2, label='$V(x) = |x|$')
line_Vx.set_data(eqn.x, eqn.V)
xdata, ydata = [], []

ax1.legend(loc=1)
ax1.set_xlabel('$x$')
ax1.set_ylabel("Energy")

# Momentum space
ymin = 0
ymax = 1/x.size * prob_tk.max()
ax2 = fig.add_subplot(212, xlim = (-2*abs(k0), 2*abs(k0)), ylim=(ymin, ymax))
line_prob_kt, =  ax2.plot([], [], 'r-', lw=2, label='$|\psi(k,t)|^2$')

ax2.legend(loc=1)
ax2.set_xlabel('$k$')
ax2.set_ylabel("Probability density")


def init():
    line_prob_xt.set_data([], [])
    line_Vx.set_data([], [])
    line_prob_kt.set_data([], [])
    return line_prob_xt, line_Vx, line_prob_kt


def update_line(i):
    probmax = np.max(np.abs(prob_tx))
    scale_factor = (Vmax-E)/probmax
    line_prob_xt.set_data(x, E + scale_factor*prob_tx[i])
    line_prob_kt.set_data(eqn.k, 1/x.size * prob_tk[i])
    line_Vx.set_data(x, V)
    return line_prob_xt, line_prob_kt, line_Vx


ani = animation.FuncAnimation(fig, update_line,
                              np.arange(0, prob_tx.shape[0]),
                              interval=15,
                              init_func=init,
                              blit=True)
FFMpegWriter = animation.writers['ffmpeg']

ani.save('animations/time_evolve.mp4',
         fps=30, dpi=120,
         extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
