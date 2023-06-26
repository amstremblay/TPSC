from GF import *
"""
Date: June 23, 2023
"""
class TPSC:
    def __init__(self, mesh, U, n):
        """
        Class to perform a TPSC calculation.
        mesh: Mesh object from the Mesh.py file
        n: input density
        """
        # Initialize the variables
        self.mesh = mesh
        self.U = U
        self.n = n

        
    def calcFirstLevelApprox(self):
        """
        Do the first level of approximation of TPSC.
        This calculates chi1, and then obtains chisp and chich from the sum rules and the TPSC ansatz.
        """
        # Calculate chi1
        self.calcChi1()

        # Calculate Usp and Uch from the TPSC ansatz
        self.calcUsp()
        self.calcUch()

        # Calculate the double occupancy
        self.docc = self.calcDoubleOccupancy()

        # Calculate the spin correlation length
        self.calcXispCommensurate()
    
    def calcChi1(self):
        """
        Function to calculate chi1(q,iqn).
        This also calculates the trace of chi1(q,iqn) as a consistency check.
        """
        # Calculate the Green function G1 at the first level of approximation of TPSC
        self.g1 = GF(self.mesh, self.n)
        self.g1.calcGtaur()
        self.g1.calcGtaumr()

        # Calculate chi1(tau,r)
        self.chi1 = 2.*self.g1.gtaur * self.g1.gtaumr[::-1, :]

        # Fourier transform to (q,iqn)
        self.chi1 = self.mesh.r_to_k(self.chi1)
        self.chi1 = self.mesh.tau_to_wn('B', self.chi1)
        
        # Calculate the trace of chi1
        chi1_trace = np.sum(self.chi1, axis=1)/self.mesh.nk
        chi1_trace_l  = self.mesh.IR_basis_set.smpl_wn_b.fit(chi1_trace)
        self.traceChi1 = self.mesh.IR_basis_set.basis_b.u(0)@chi1_trace_l
    
    def calcUsp(self):
        """
        Function to compute Usp from chi1 and the sum rule.
        """
        # Bounds on the value of Usp
        Uspmin = 0.
        Uspmax = 2./np.amax(self.chi1).real-1e-7 # Note: the 1e-7 is chosen for stability purposes

        # Calculate Usp
        self.Usp = brentq(lambda m: self.calcSumChisp(m)-self.calcSumRuleChisp(m), Uspmin, Uspmax, disp=True) 

    def calcUch(self, Uchmin=0., Uchmax=100.):
        """
        Function to compute Uch from chi1 and the sum rule.
        Note: calcUsp has to be called before this function.
        """
        # Calculate Uch
        self.Uch = brentq(lambda m: self.calcSumChich(m)-self.calcSumRuleChich(self.Usp), Uchmin, Uchmax, disp=True) 

    def calcSumChisp(self, Usp):
        """
        Function to compute the trace of chisp(q) = chi1(q)/(1-Usp/2*chi1(q)).
        Also sets chisp(q,iqn) and finds the maximal value.
        """
        self.chisp = self.chi1/(1-0.5*Usp*self.chi1)
        self.chispmax = np.amax(self.chisp)
        chisp_trace = np.sum(self.chisp, axis=1)/self.mesh.nk
        chisp_trace_l  = self.mesh.IR_basis_set.smpl_wn_b.fit(chisp_trace)
        chisp_trace = self.mesh.IR_basis_set.basis_b.u(0)@chisp_trace_l

        return chisp_trace.real

    def calcSumChich(self, Uch):
        """
        Function to compute the trace of chich(1) = chi1(q)/(1+Uch/2*chi1(q)).
        Also sets chich(q,iqn).
        """
        self.chich = self.chi1/(1+0.5*Uch*self.chi1)
        chich_trace = np.sum(self.chich, axis=1)/self.mesh.nk
        chich_trace_l  = self.mesh.IR_basis_set.smpl_wn_b.fit(chich_trace)
        chich_trace = self.mesh.IR_basis_set.basis_b.u(0)@chich_trace_l

        return chich_trace.real

    def calcDoubleOccupancy(self):
        """
        Function to compute the double occupancy.
        Note: the function calcUsp has to be called before this one
        The TPSC ansatz we use here satisfies the particle-hole symmetry with:
        n<1: Usp = U<n_up n_dn>/(<n_up><n_dn>)
        n>1: Usp = U<(1-n_up)(1-n_dn)>/(<(1-n_up)><(1-n_dn)>)
        """
        if (self.n<1):
            return self.Usp/self.U*self.n*self.n/4
        else:
            return self.Usp/(4*self.U)*(2-self.n)*(2-self.n)-1+self.n

    def calcSumRuleChisp(self, Usp):
        """
        Calculate the spin susceptibility sum rule for a specific Usp and U.
        The TPSC ansatz we use here satisfies the particle-hole symmetry with:
        n<1: Usp = U<n_up n_dn>/(<n_up><n_dn>)
        n>1: Usp = U<(1-n_up)(1-n_dn)>/(<(1-n_up)><(1-n_dn)>)
        """
        if self.n<1:
            return self.n - Usp/self.U*self.n*self.n/2
        else:
            return self.n - Usp/(2*self.U)*(2-self.n)*(2-self.n)+2-2*self.n
    
    def calcSumRuleChich(self, Usp):
        """
        Calculate the charge susceptibility sum rule for a specific Usp and U.
        The TPSC ansatz we use here satisfies the particle-hole symmetry with:
        n<1: Usp = U<n_up n_dn>/(<n_up><n_dn>)
        n>1: Usp = U<(1-n_up)(1-n_dn)>/(<(1-n_up)><(1-n_dn)>)
        """
        if self.n<1:
            return self.n + Usp/self.U*self.n*self.n/2 - self.n*self.n
        else:
            return self.n + Usp/(2*self.U)*(2-self.n)*(2-self.n)-2+2*self.n - self.n*self.n
    
    def calcXispCommensurate(self):
        """
        Compute the spin correlation length from commensurate spin fluctuations at Q=(pi,pi).
        This calculates the width at half maximum of the spin susceptibility ONLY if its maximal value is at (pi,pi).
        If the spin susceptibility maximum is not at (pi,pi) (incommensurate spin fluctuations), this function returns -1.
        """
        # Set the default value
        self.xisp = -1
        qy = 0
        qx = int(self.mesh.nk1/2)
        qHM = 0
        q0 = 0
        index_peak = np.argmax(self.chisp[self.mesh.iw0_b])
        # Calculate the spin susceptibility from commensurate fluctuations
        if (index_peak == int(self.mesh.nk1*qx+qx)):
            chispmax = self.chisp[self.mesh.iw0_b, index_peak].real
            chisphalf = self.chisp[self.mesh.iw0_b,int(self.mesh.nk1*qx)+qy].real
            chisptemp = chisphalf
            while (chisphalf < chispmax/2 and qy < self.mesh.nk1/2):
                chisptemp = chisphalf
                qy = qy+1
                chisphalf = self.chisp[self.mesh.iw0_b, int(self.mesh.nk1*qx)+qy].real
            if qy>0: 
                q0 = 2*np.pi*(qy-1)/self.mesh.nk1
                qHM = 2*np.pi/self.mesh.nk1*(chispmax/2 - chisptemp)/(chisphalf - chisptemp)
            self.xisp = 1/(np.pi - qHM - q0)
    
    def calcSecondLevelApprox(self):
        """
        Function to calculate the self-energy in the second level of approximation of TPSC.
        Important: The function calcFirstLevelApprox must be called before this one.
        Note: The Hartree term (Un/2) is not included here.
        The TPSC self-energy is: U/8\sum_q(3chi_sp(q)U_sp + chi_ch(q)U_ch)G1(k+q).
        We define V(q) =  U/8(3chi_sp(q)U_sp + chi_ch(q)U_ch) and compute 1/2(V(r)*G(-r)+V(-r)G(r)).
        """
        # Get V(iqn,q) 
        V = self.U/8*(3.*self.Usp*(self.chisp)+self.Uch*(self.chich)) 

        # Get V(tau,r)
        Vp = self.mesh.k_to_r(V)
        Vm = self.mesh.k_to_mr(V)
        Vp = self.mesh.wn_to_tau('B', Vp)
        Vm = self.mesh.wn_to_tau('B', Vm)

        # Calculate the self-energy in (r,tau) space
        self.selfEnergy = 0.5*(Vm*self.g1.gtaur+Vp*self.g1.gtaumr)

        # Fourier transform
        self.selfEnergy = self.mesh.r_to_k(self.selfEnergy)
        self.selfEnergy = self.mesh.tau_to_wn('F', self.selfEnergy)

        # Calculate G2
        self.g2 = GF(self.mesh, self.n, self.selfEnergy)
        self.g2.calcGtaur()
        self.g2.calcGtaumr()

    
    def checkSelfConsistency(self):
        """
        Function to check the self-consistency between one- and two-particle quantities through:
        Tr[Self-Energy*Green's function] = U<n_up n_dn> - Un^2/4
        The -Un^2/4 term on the right hand side is due to the fact that the Hartree term is not included in the self-energy.
        In TPSC, the self-consistency check is exact when computed with the Green's function at the first level of approximation,
        but it is not with the Green's function G2. The discrepancy between the exact result and the trace with G2 is 
        a check of the validity of the TPSC calculation.
        """
        # Calculate the traces
        self.calcTraceSelfG(level=1)
        self.calcTraceSelfG(level=2)

        # Calculate the expected result
        self.exactTraceSelfG = self.U*self.docc-self.U*self.n*self.n/4
    
    def calcTraceSelfG(self,level):
        """
        Calculate the trace of Self-Energy*G^(level)
        level: level of approximation for the Green's function used in the calculation, either 1 or 2
        Note: functions to calculate first and second levels of approximation must be called before this one
        """
        if level==1:
            # Calculate the trace of Self-Energy*G1
            trace = np.sum(self.selfEnergy*self.g1.giwnk, axis=1)/self.mesh.nk
            trace_l  = self.mesh.IR_basis_set.smpl_wn_f.fit(trace)
            self.traceSG1 = self.mesh.IR_basis_set.basis_f.u(0)@trace_l
        elif level==2:
            # Calculate the trace of Self-Energy*G2
            trace = np.sum(self.selfEnergy*self.g2.giwnk, axis=1)/self.mesh.nk
            trace_l  = self.mesh.IR_basis_set.smpl_wn_f.fit(trace)
            self.traceSG2 = self.mesh.IR_basis_set.basis_f.u(0)@trace_l
            

    







