import TPSC
import json

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

tpsc = TPSC.TPSC(parameters)
out = tpsc.run()
tpsc.printResults()

#with open("main_results.json", 'w') as outfile:
#    outfile.write(json.dumps(out, indent=4))
