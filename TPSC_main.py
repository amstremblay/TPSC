from Dispersions import *
import sparse_ir
from TPSC import *

"""
Date: June 26, 2023
"""

# Calculation parameters
# TODO fichier de paramètres
n = 1.2      # Density
T = 0.4    # Temperature     
t = 1.        # First neighbour hopping
tp = 0      # Second neighbour hopping
tpp = 0
nkx = 64    # Number of k-points in one space direction
nk = nkx*nkx    # Total number of k-points
W = 9           # Bandwidth (inutile, juste pour se souvenir qu'il faut donner une valeur raisonnable à wmax)
U = 3.5

# IR basis set 
wmax = 10       # Has to be larger than W
IR_tol    = 1e-10
IR_basis_set = sparse_ir.FiniteTempBasisSet(1./T, wmax, eps=IR_tol)

# Generate the dispersion and the mesh
k1, k2 = np.meshgrid(np.arange(nkx)/nkx, np.arange(nkx)/nkx)    # k-point grid
dispersion = calcDispersion2DSquare(k1, k2, nk, t=t, tp=tp, tpp=tpp)
mesh = Mesh2D(IR_basis_set, nkx, nkx, T, dispersion)

# Do the TPSC calculation
tpsc = TPSC(mesh, U, n)
tpsc.calcFirstLevelApprox()
tpsc.calcSecondLevelApprox()
tpsc.checkSelfConsistency()

print(f"Trace of chi1: {tpsc.traceChi1:.5f}")
print(f"Usp: {tpsc.Usp:.5f}")
print(f"Uch: {tpsc.Uch:.5f}")
print(f"Trace Self*G1: {tpsc.traceSG1:.5f}")
print(f"Trace Self*G2: {tpsc.traceSG2:.5f}")
print(f"U*Double occupation - U*density^2/4: {tpsc.exactTraceSelfG:.5f}")
print(f"mu1: {tpsc.g1.mu:.5f}")
print(f"mu2: {tpsc.g2.mu:.5f}")

