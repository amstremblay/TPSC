from TPSC import *
from TPSC.Dispersions import *
from TPSC.TPSC import *

import sparse_ir
import numpy as np
import matplotlib.pyplot as plt
import sys
import json

"""
Date: July 13, 2023
This script runs a TPSC calculation (from a given para.json parameters file) and plots the self-energy
as a function of w_n and k.
"""

parameters = {
    "dispersion" : "square",  # Dispersion model
    "t" : 1,                  # First neighbour hopping
    "tp" : 1,                 # Second neighbour hopping
    "tpp" : 0,                # Third neighbour hopping
    "T" : 0.1,                # Temperature
    "U" : 2.0,                # ?
    "n" : 1,                  # Density
    "nkx" : 64,               # Number of k-points in one space direction
    "wmax_mult" : 1.25,       # for IR basis, multiple of bandwidth to use as wmax (must be greater than 1)
    "IR_tol" : 1e-12          # # for IR basis, tolerance of intermediate representation
}

# Do the TPSC calculation
tpsc = TPSC(parameters)
tpsc.run()
tpsc.printResults()
mesh = tpsc.mesh

# EXAMPLE 1 - EVALUATING THE SELF-ENERGY AS A FUNCTION OF WN AT A SPECIFIC K-POINT (pi/2, pi/2)
tpsc.plotSelfEnergyVsWn((np.pi/2, np.pi/2), (-50,50), show=True)
exit()

# EXAMPLE 2 - EXTRAPOLATING THE SELF-ENERGY AT w_n -> 0 as a function of k
selfen_zerofreq = mesh.extrapolate_zero_freq(tpsc.selfEnergy, 5).reshape((mesh.nk1, mesh.nk2))
print(selfen_zerofreq.shape)

plt.figure()
plt.contourf(2*np.pi*mesh.k1, 2*np.pi*mesh.k2, selfen_zerofreq.real, levels=25, cmap='magma')
plt.xlabel(r"$k_x$")
plt.ylabel(r"$k_y$")
plt.colorbar()
plt.title(r"real part, extrapolation of self at $\omega_n \to 0$")


plt.figure()
plt.contourf(2*np.pi*mesh.k1, 2*np.pi*mesh.k2, selfen_zerofreq.imag, levels=25, cmap='magma')
plt.xlabel(r"$k_x$")
plt.ylabel(r"$k_y$")
plt.colorbar()
plt.title(r"imag part, extrapolation of self at $\omega_n \to 0$")

plt.show()
