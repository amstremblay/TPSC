import sparse_ir

from TPSC import *
from TPSC.Dispersions import *
import sys
import json

"""
Date: June 26, 2023
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




