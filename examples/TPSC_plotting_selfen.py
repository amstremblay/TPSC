from Dispersions import *
import sparse_ir
from TPSC import *
import sys
import json

"""
Date: July 13, 2023
This script runs a TPSC calculation (from a given parameters file) and plots the self-energy
as a function of w_n and k. 
Note: This script must be run from the base TPSC folder, using for example the command
"python3 examples/TPSC_plotting_selfen.py para.json"
"""


# We read the parameters file, passed as a command line argument
if len(sys.argv) == 2:
    para_filename = sys.argv[1]
else:
    para_filename = "para.json"

with open(para_filename, 'r') as infile:
    params = json.load(infile)

# Calculation parameters
lattice_dispersion = params["dispersion"].lower() # dispersion relation to use (square/triangular)
n = params["n"]     # Density
T = params["T"]   # Temperature     
t = params["t"]        # First neighbour hopping
tp = params["tp"]     # Second neighbour hopping
tpp = params["tpp"]   # Third neighbour hopping
nkx = params["nkx"]    # Number of k-points in one space direction
nk = nkx*nkx    # Total number of k-points
U = params["U"]

wmax_mult = params["wmax_mult"]  # for IR basis, multiple of bandwidth to use as wmax (must be greater than 1)
IR_tol = params["IR_tol"]        # for IR basis, tolerance of intermediate representation

# Generate the dispersion and the k-mesh
k1, k2 = np.meshgrid(np.arange(nkx)/nkx, np.arange(nkx)/nkx)    # k-point grid
if lattice_dispersion == "square":
    dispersion = calcDispersion2DSquare(k1, k2, nk, t=t, tp=tp, tpp=tpp)
elif lattice_dispersion == "triangle":
    dispersion = calcDispersion2DTriangle(k1, k2, nk, t=t, tp=tp)

# Compute the bandwidth and define the IR basis 
W = dispersion.max() - dispersion.min()
wmax = W * wmax_mult      
IR_basis_set = sparse_ir.FiniteTempBasisSet(1./T, wmax, eps=IR_tol)

# Define the meshes used in the TPSC calculation
mesh = Mesh2D(IR_basis_set, nkx, nkx, T, dispersion)

# Do the TPSC calculation
tpsc = TPSC(mesh, U, n)
tpsc.calcFirstLevelApprox()
tpsc.calcSecondLevelApprox()
tpsc.checkSelfConsistency()


# Print the main results on screen
print(f"Trace of chi1: {tpsc.traceChi1:.5f}")
print(f"Usp: {tpsc.Usp:.5f}")
print(f"Uch: {tpsc.Uch:.5f}")
print(f"doubleocc : {tpsc.docc:.5f}")
print(f"Trace Self*G1: {tpsc.traceSG1:.5f}")
print(f"Trace Self*G2: {tpsc.traceSG2:.5f}")
print(f"U*Double occupation - U*density^2/4: {tpsc.exactTraceSelfG:.5f}")
print(f"mu1: {tpsc.g1.mu:.5f}")
print(f"mu2: {tpsc.g2.mu:.5f}")

# Produce the output file with the same main results
main_results = {}
main_results["Usp"] = tpsc.Usp
main_results["Uch"] = tpsc.Uch
main_results["doubleocc"] = tpsc.docc
main_results["Trace_chi1"] = [tpsc.traceChi1.real, tpsc.traceChi1.imag]
main_results["Trace_Self2_G1"] = [tpsc.traceSG1.real, tpsc.traceSG1.imag]
main_results["Trace_Self2_G2"] = [tpsc.traceSG2.real, tpsc.traceSG2.imag]
main_results["U*doubleocc-U*n/4"] = tpsc.exactTraceSelfG
main_results["mu1"] = tpsc.g1.mu
main_results["mu2"] = tpsc.g2.mu

with open("main_results.json", 'w') as outfile:
    outfile.write(json.dumps(main_results, indent=4))



######### PLOTTING THE SELF-ENERGY ###########
import matplotlib.pyplot as plt


# EXAMPLE 1 - EVALUATING THE SELF-ENERGY AS A FUNCTION OF WN AT A SPECIFIC K-POINT (pi/2, pi/2)
inds = np.arange(-50, 50.01, 1)
ind_kpoint_node = mesh.get_ind_kpt(np.pi/2, np.pi/2)
vals_s2 = mesh.get_specific_wn("F", tpsc.selfEnergy[:,ind_kpoint_node], inds)
plt.figure()
plt.title("self-energy node")
plt.plot(inds, vals_s2.real, 'b', label="Re")
plt.plot(inds, vals_s2.imag, 'r', label="Im")
plt.xlabel(r"$n$")
plt.ylabel(r"$\Sigma$")
plt.legend(loc='best')

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