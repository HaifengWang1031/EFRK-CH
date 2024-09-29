from typing import Tuple
from time import perf_counter
from functools import partial

import numpy as np
import scipy.fft as fft
import numpy.typing as tnp

class Gradient_Flow_Solver():

    def linear_operator(self):
        pass    

    def nonlinear_operator(self):
        pass    

    def init_record(self):
        pass

    def result_record(self,Un,i) -> str:
        pass
    
    def __init__(self,step_method) -> None:
        self.step_method = step_method

        # Step Method
        ## Explicit forward Euler method (FE)
        if step_method == "FE":
            self.step = self.FE
        ## Implicit Explicit method (IMEX)
        elif step_method == "IMEX":
            self.step = self.IMEX
        ## Integrating factor Runge--Kutta method (IFRK)
        elif step_method == "IFRK1":
            self.step = self.IFRK1
        elif step_method == "IFRK2":
            self.step = self.IFRK2
        elif step_method == "IFRK3_Heun":
            self.step = self.IFRK3_Heun
        ## Exponential free Runge--Kutta method (EFRK)
        elif step_method == "EFRK1":
            self.step = self.IMEX
        elif step_method == "EFRK2":
            self.step = self.EFRK2
        elif step_method == "EFRK3_Heun":
            self.step = self.EFRK3_Heun
        ## Operator Splitting method
        elif step_method == "Lie_Trotter":
            self.step = self.Lie_Trotter
        elif step_method == "StrangSplitting":
            self.step = self.StrangSplitting
        ## Exponential time difference method (ETD)
        elif step_method == "ETD1":
            self.step = self.ETD1
        elif step_method == "ETDRK2":
            self.step = self.ETDRK2
        elif step_method == "ETDRK3":
            self.step = self.ETDRK3

    def FE(self,Un,tn):
        self.phi1 = 1 + self.tau * self.Lk

        fUn = self.ft(Un)
        fUn1 = self.phi1*fUn + self.tau*self.Nk(fUn,tn)
        return self.ift(fUn1).real

    def IMEX(self,Un,tn):
        self.phi1 = 1 - self.tau*self.Lk

        fUn = self.ft(Un)
        fUn1 = 1/self.phi1*(fUn + self.tau*self.Nk(fUn,tn))
        return self.ift(fUn1).real

    def IFRK1(self,Un,tn):
        self.expM = np.exp(self.tau*self.Lk)

        fUn = self.ft(Un)
        fUn1 = self.expM * fUn + self.tau * self.expM * self.Nk(fUn,tn)
        return self.ift(fUn1).real

    def IFRK2(self,Un,tn):
        a10 = 1
        a20,a21 = 1/2,1/2

        self.expM = np.exp(self.tau * self.Lk)
        
        fUn = self.ft(Un)
        fUn1 = self.expM*fUn + self.tau * (a10 * self.expM * self.Nk(fUn,tn))
        fUn2 = self.expM*fUn + self.tau * (a20 * self.expM * self.Nk(fUn,tn) + a21 * self.Nk(fUn1,tn+a10*self.tau))

        return self.ift(fUn2).real
        
    def IFRK3_Heun(self,Un,tn):
        a10 = 1/3
        a20,a21 = 0,2/3
        a30,a31,a32 = 1/4,0,3/4

        self.expM13 = np.exp(1/3*self.tau*self.Lk)
        self.expM23 = np.exp(2/3*self.tau*self.Lk)
        self.expM   = np.exp(self.tau*self.Lk)

        fUn  = self.ft(Un)
        fUn1 = self.expM13*fUn + self.tau * (a10 * self.expM13 * self.Nk(fUn,tn))
        fUn2 = self.expM23*fUn + self.tau * (a20 * self.expM23 * self.Nk(fUn,tn) + a21 * self.expM13 * self.Nk(fUn1,tn+a10*self.tau))
        fUn3 = self.expM*fUn   + self.tau * (a30 * self.expM   * self.Nk(fUn,tn) + a31 * self.expM23 * self.Nk(fUn1,tn+a10*self.tau) + a32 * self.expM13 * self.Nk(fUn2,tn+(a20+a21)*self.tau))

        return self.ift(fUn3).real

    def EFRK2(self,Un,tn):
        a10 = 1
        a20,a21 = 1/2,1/2

        self.phi0 = np.ones_like(self.Lk)
        self.phi1 = self.phi0 - self.tau * (a10 * self.phi0 * self.Lk)
        self.phi2 = self.phi0 - self.tau * (a20 * self.phi0 * self.Lk + a21 * self.phi1 *self.Lk)
        
        fUn = self.ft(Un)
        Nk1 = self.Nk(fUn,tn)
        fUn1 = 1/self.phi1*(fUn + self.tau * (a10 * self.phi0*Nk1))
        fUn2 = 1/self.phi2*(fUn + self.tau * (a20 * self.phi0*Nk1 + a21 * self.phi1 * self.Nk(fUn1,tn+a10*self.tau)))

        return self.ift(fUn2).real

    def EFRK3_Heun(self,Un,tn):
        a10 = 1/3
        a20,a21 = 0,2/3
        a30,a31,a32 = 1/4,0,3/4

        self.phi0 = np.ones_like(self.Lk)
        self.phi1 = self.phi0 - self.tau * (a10 * self.phi0 * self.Lk)
        self.phi2 = self.phi0 - self.tau * (a20 * self.phi0 * self.Lk + a21 * self.phi1 *self.Lk)
        self.phi3 = self.phi0 - self.tau * (a30 * self.phi0 * self.Lk + a31 * self.phi1 *self.Lk + a32 * self.phi2 *self.Lk)

        fUn = self.ft(Un)
        Nk1 = self.Nk(fUn,tn)
        fUn1 = 1/self.phi1*(fUn + self.tau * (a10 * self.phi0 * Nk1))
        fUn2 = 1/self.phi2*(fUn + self.tau * (a20 * self.phi0 * Nk1 + a21 * self.phi1 * self.Nk(fUn1,tn+a10*self.tau)))
        fUn3 = 1/self.phi3*(fUn + self.tau * (a30 * self.phi0 * Nk1 + a31 * self.phi1 * self.Nk(fUn1,tn+a10*self.tau) + a32 * self.phi2 * self.Nk(fUn2,tn+(a20+a21)*self.tau)))

        return self.ift(fUn3).real

    def ETD1(self,Un,tn):
        M = 32
        dim = self.Lk.ndim
        r  = np.expand_dims(np.exp( 1j*np.pi*(np.arange(1,M+1) - .5)/M ),axis = list(range(dim)) )
        Lr = self.tau*np.expand_dims(self.Lk,axis=-1) + r
        self.phi0 = np.exp(self.tau*self.Lk)
        self.phi1 = np.mean((1-np.exp(Lr))/(-Lr),axis=-1).real 

        fUn = self.ft(Un)
        fUn1 = self.phi0*fUn + self.tau*self.phi1*self.Nk(fUn,tn)

        return self.ift(fUn1).real

    def ETDRK2(self,Un,tn):

        M = 32
        dim = self.Lk.ndim
        r  = np.expand_dims(np.exp( 1j*np.pi*(np.arange(1,M+1) - .5)/M ),axis = list(range(dim)) )
        Lr = self.tau*np.expand_dims(self.Lk,axis=-1) + r
        self.phi0 = np.exp(self.tau*self.Lk)
        self.phi1 = np.mean((1-np.exp(Lr))/(-Lr),axis=-1).real
        self.phi2 = np.mean((-Lr - 1 + np.exp(Lr))/(-Lr)**2,axis=-1).real

        fUn =  self.ft(Un)
        Nk1 = self.Nk(fUn,tn)

        fUn1 = self.phi0*fUn + self.tau*self.phi1*Nk1
        fUn2 = self.phi0*fUn + self.tau*((self.phi1 - self.phi2)*Nk1 + self.phi2*self.Nk(fUn1,tn+self.tau))

        return self.ift(fUn2).real

    def ETDRK3(self,Un,tn,c1 = 4/9):

        c2 = 2/3

        if not hasattr(self,"phi01"):
            M = 32
            dim = self.Lk.ndim
            r  = np.expand_dims(np.exp( 1j*np.pi*(np.arange(1,M+1) - .5)/M ),axis = list(range(dim)) )
            Lr = self.tau*np.expand_dims(self.Lk,axis=-1) + r

            self.phi01 = np.exp(c1*self.tau*self.Lk)
            self.phi02 = np.exp(c2*self.tau*self.Lk)
            self.phi0  = np.exp(self.tau*self.Lk)

            self.phi11 = np.mean((1-np.exp(c1*Lr))/(-c1*Lr),axis=-1).real
            self.phi12 = np.mean((1-np.exp(c2*Lr))/(-c2*Lr),axis=-1).real
            self.phi1 = np.mean((1-np.exp(Lr))/(-Lr),axis=-1).real

            self.phi22 = np.mean((-c2*Lr - 1 + np.exp(c2*Lr))/(-c2*Lr)**2,axis=-1).real
            self.phi2  = np.mean((-Lr - 1 + np.exp(Lr))/(-Lr)**2,axis=-1).real

        fUn =  self.ft(Un)
        fUn1 = self.phi01*fUn + self.tau*(c1*self.phi11*self.Nk(fUn,tn))
        fUn2 = self.phi02*fUn + self.tau*((2/3*self.phi12 - 4/(9*c1)*self.phi22)*self.Nk(fUn,tn) \
                                         + 4/(9*c1)*self.phi22*self.Nk(fUn1,tn+c1*self.tau))
        fUn3 = self.phi0*fUn  + self.tau*( (self.phi1 - 3/2*self.phi2) *self.Nk(fUn,tn) \
                                        + 3/2*self.phi2*self.Nk(fUn2,tn+c2*self.tau))

        return self.ift(fUn3).real

    def Lie_Trotter(self,Un,tn):
        self.Sl = np.exp(self.tau * self.Lk)

        fUn  = self.ft(Un)
        fUn1 = self.Sl * fUn

        fUn2 = fUn1 + self.tau * self.Nk(fUn1,tn)
        return self.ift(fUn2).real

    def StrangSplitting(self,Un,tn):
        self.Sl = np.exp(1/2*self.tau*self.Lk)

        fUn  = self.ft(Un)
        fUn1 = self.Sl*fUn

        fUn21 = fUn1 + self.tau * self.Nk(fUn1,tn)
        fUn22 = fUn1 + self.tau * 1 / 2 * self.Nk(fUn21,tn + 1 / 2*self.tau)

        fu3 = self.Sl * fUn22
        return self.ift(fu3).real

    def solve(self,t_span,tau):
        [self.T0, self.T] = t_span; self.tau = tau
        M = int((self.T-self.T0)/tau)
        self.tn = np.linspace(self.T0,self.T, M+1)

        # Numerical solution
        self.Un = np.empty([len(self.tn)]+list(self.U0.shape)); self.Un[0] = self.U0
        # Correspond property
        self.init_record()
        self.time_elapse = np.zeros(M+1)

        Un = self.Un[0]
        for i,tn in enumerate(self.tn[1:]):
            start_time = perf_counter()
            Un = self.step(Un,tn-self.tau)
            end_time = perf_counter()

            cpu_time = end_time - start_time
            self.time_elapse[i+1] = self.time_elapse[i] + cpu_time

            self.Un[i+1] = Un

            msg = "."
            msg = self.result_record(Un,i)

            print(f"\r {tn:.2f}\\{self.T}, Elapse: {end_time - start_time:.3f} s, "+ msg+" "*10, end="")
        print("")

    def solve_long_time(self,t_span,tau,snapshot_time = []):
        [self.T0, self.T] = t_span; self.tau = tau
        M = int((self.T-self.T0)/tau)
        self.tn = np.linspace(self.T0,self.T, M+1)

        # Numerical solution
        snapshot_time = np.sort(snapshot_time)
        if len(snapshot_time) == 0:
            self.Un = np.empty([int(self.T - self.T0)+1]+list(self.U0.shape)); self.Un[0] = self.U0
        else:
            self.Un = np.empty([len(snapshot_time)]+list(self.U0.shape)); 
            if np.isclose(np.min(np.abs(self.tn[0] - snapshot_time)),0,atol=1e-10):
                index = np.argmin(np.abs(self.tn[0] - snapshot_time))
                self.Un[index] = self.U0

        # Correspond property
        self.init_record()
        self.time_elapse = np.zeros(M+1)

        Un = self.U0
        for i,tn in enumerate(self.tn[1:]):

            start_time = perf_counter()
            Un = self.step(Un,tn)
            end_time = perf_counter()

            cpu_time = end_time - start_time
            self.time_elapse[i+1] = self.time_elapse[i] + cpu_time
            
            if len(snapshot_time) == 0:
                if tn%1 == 0: 
                    self.Un[int(tn - self.T0)] = Un
            else:
                if np.isclose(np.min(np.abs(tn - snapshot_time)),0,atol=1e-10):
                    index = np.argmin(np.abs(tn - snapshot_time))
                    self.Un[index] = Un

            msg = self.result_record(Un,i)

            print(f"\r {tn:.2f}\\{self.T}, Elapse: {cpu_time:.3f} s, "+ msg+" "*5, end="")
        print("")

class Ginzburg_Landau_type_Solver(Gradient_Flow_Solver):
    def __init_1D(self,s_domain,discrete_num,initial_condition):
        self.xa, self.xb = s_domain
        self.Nx, = discrete_num
        self.hx = (self.xb - self.xa) / self.Nx
        self.h = self.hx

        self.xn = np.linspace(self.xa, self.xb, self.Nx + 1)

        self.ft = fft.fft
        self.ift = fft.ifft
        
        # presudo spectral method
        self.mu_x = 2 * np.pi / (self.xb - self.xa)

        
        self.D_x = np.fft.fftshift(1j*self.mu_x*np.array([0]+ [i for i in range(-self.Nx//2+1,self.Nx//2)]))
        self.D_xx = self.D_x**2

        self.Lap = self.D_xx

        # Linear Operator 
        self.Lk = self.linear_operator()
        # Nonlinear Operator
        self.Nk = lambda fu,t: self.nonlinear_operator(fu)

        self.U0 = initial_condition[:-1]
        self.E0 = self.energy(self.U0)
        self.M0 = self.mass(self.U0)

    def __init_2D(self,s_domain,discrete_num,initial_condition):
        # spatial discrete
        self.xa, self.ya, self.xb, self.yb = s_domain
        self.Nx, self.Ny = discrete_num
        self.hx = (self.xb - self.xa) / self.Nx
        self.hy = (self.yb - self.ya) / self.Ny
        self.h = self.hx*self.hy

        self.xn = np.linspace(self.xa, self.xb, self.Nx + 1)
        self.yn = np.linspace(self.ya, self.yb, self.Ny + 1)
        self.X,self.Y = np.meshgrid(self.xn,self.yn)

        self.ft = fft.fft2
        self.ift = fft.ifft2
        
        # presudo spectral method
        self.mu_x = 2 * np.pi / (self.xb - self.xa)
        self.mu_y = 2 * np.pi / (self.yb - self.ya)

        self.D_x = np.fft.fftshift(1j*self.mu_x*np.array([0]+ [i for i in range(-self.Nx//2+1,self.Nx//2)]))[np.newaxis,:]
        self.D_y = np.fft.fftshift(1j*self.mu_y*np.array([0]+ [i for i in range(-self.Ny//2+1,self.Ny//2)]))[:,np.newaxis]
        self.D_xx = self.D_x**2
        self.D_yy = self.D_y**2

        self.Lap = self.D_xx + self.D_yy

        # Linear Operator 
        self.Lk = self.linear_operator()
        # Nonlinear Operator
        self.Nk = lambda fu,t: self.nonlinear_operator(fu)

        self.U0 = initial_condition[:-1,:-1]
        self.E0 = self.energy(self.U0)
        self.M0 = self.mass(self.U0)

    def __init_3D(self,s_domain,discrete_num,initial_condition):
        # spatial discrete
        self.xa, self.ya, self.za, self.xb, self.yb, self.zb = s_domain
        self.Nx, self.Ny, self.Nz = discrete_num
        self.hx = (self.xb - self.xa) / self.Nx
        self.hy = (self.yb - self.ya) / self.Ny
        self.hz = (self.zb - self.za) / self.Nz
        self.h = self.hx*self.hy*self.hz

        self.xn = np.linspace(self.xa, self.xb, self.Nx + 1)
        self.yn = np.linspace(self.ya, self.yb, self.Ny + 1)
        self.zn = np.linspace(self.za, self.zb, self.Nz + 1)
        self.X,self.Y,self.Z = np.meshgrid(self.xn,self.yn,self.zn)

        self.ft = fft.fftn
        self.ift = fft.ifftn

        # presudo spectral method
        self.mu_x = 2 * np.pi / (self.xb - self.xa)
        self.mu_y = 2 * np.pi / (self.yb - self.ya)
        self.mu_z = 2 * np.pi / (self.zb - self.za)

        self.D_x = np.fft.fftshift(1j*self.mu_x*np.array([0] + [i for i in range(-self.Nx//2+1,self.Nx//2)]))[np.newaxis,:,np.newaxis]
        self.D_y = np.fft.fftshift(1j*self.mu_y*np.array([0] + [i for i in range(-self.Ny//2+1,self.Ny//2)]))[:,np.newaxis,np.newaxis]
        self.D_z = np.fft.fftshift(1j*self.mu_z*np.array([0] + [i for i in range(-self.Nz//2+1,self.Nz//2)]))[np.newaxis,np.newaxis,:]
        self.D_xx = self.D_x**2
        self.D_yy = self.D_y**2
        self.D_zz = self.D_z**2

        self.Lap = self.D_xx + self.D_yy + self.D_zz

        # Linear Operator 
        self.Lk = self.linear_operator()
        # Nonlinear Operator
        self.Nk = lambda fu,t: self.nonlinear_operator(fu)

        self.U0 = initial_condition[:-1,:-1,:-1]
        self.E0 = self.energy(self.U0)
        self.M0 = self.mass(self.U0)

    def F(self,u):
        """
        $F(u) = \frac{1}{4}(u^2 - 1)^2
        """
        return 1/4*(u**2 - 1)**2

    def f(self,fu):
        """
        f(u) = u**3 - u
        """
        u = self.ift(fu).real
        return self.ft(u**3 - u)
    
    def gradient_flow_L2(self):
        return -1
    
    def gradient_flow_Hm1(self):
        return self.Lap

    def linear_operator(self):
        G = self.negative_operator()
        return G*(-self.eps_sq*self.Lap + self.kap)

    def nonlinear_operator(self,fu):
        G = self.negative_operator()
        return G*(self.f(fu) - self.kap*fu)

    def energy(self,u):
        return self.h * (-self.eps_sq/2 * (self.ift(self.Lap*self.ft(u)).real).flatten() @ u.flatten() + self.F(u).flatten() @ np.ones_like(u.flatten()))
    
    def mass(self,u):
        return self.h * np.sum(u)

    def init_record(self):
        # Correspond property
        self.En =  np.empty(len(self.tn)); self.En[0] = self.E0
        self.Mn =  np.empty(len(self.tn)); self.Mn[0] = self.M0

    def result_record(self,Un,i):
        self.En[i+1] = self.energy(Un)
        self.Mn[i+1] = self.mass(Un)
        msg = f"Energy:{self.En[i+1]:.4f}, Mass:{self.Mn[i+1]:.4f}, Maximum:{np.max(Un):.2f}"
        return msg
    
    def __init__(
            self,
            gf_type: str,
            eps_sq: float,
            kappa: float,
            s_domain: Tuple[float,float ,float, float],
            discrete_num: Tuple[int,int],
            initial_condition: tnp.NDArray,
            step_method: str) -> None:

        # Step Method
        super().__init__(step_method)

        # viscous coefficient
        self.eps_sq = eps_sq
        # stabilitizer constant
        self.kap = kappa

        # The gradient flow type
        if gf_type == "L2":
            self.negative_operator = self.gradient_flow_L2
        elif gf_type == "Hm1":
            self.negative_operator = self.gradient_flow_Hm1
        
        # problem's dimension
        if len(s_domain) == 2 and len(discrete_num) == 1:
            self.__init_1D(s_domain,discrete_num,initial_condition)
        elif len(s_domain) == 4 and len(discrete_num) == 2:
            self.__init_2D(s_domain,discrete_num,initial_condition)
        elif len(s_domain) == 6 and len(discrete_num) == 3:
            self.__init_3D(s_domain,discrete_num,initial_condition)
            
    def solve_adaptive(self, t_span, tau_min, tau_max, alpha):
        [self.T0, self.T] = t_span
        M = int((self.T-self.T0)/tau_min)

        self.tau_list = np.empty(M+1); self.tau_list[0] = tau_min
        self.tn  = np.empty(M+1); self.tn[0] = self.T0

        # Numerical solution
        self.Un = np.empty([M+1]+list(self.U0.shape)); self.Un[0] = self.U0
        # Property
        self.En =  np.empty(M+1); self.En[0] = self.E0
        self.Mn =  np.empty(M+1); self.Mn[0] = self.M0
        self.time_elapse = np.zeros(M+1); self.time_elapse[0] = 0

        Un = self.Un[0]

        cur_index = 0
        while self.tn[cur_index] <= self.T:
            # current time step size
            self.tau = self.tau_list[cur_index]

            start_time = perf_counter()
            Un = self.step(Un,self.tn[cur_index])
            end_time = perf_counter()
            cpu_time = end_time - start_time

            self.Un[cur_index+1] = Un
            self.tn[cur_index+1] = self.tn[cur_index] + self.tau
            self.time_elapse[cur_index+1] = self.time_elapse[cur_index] + cpu_time

            msg = "."
            msg = self.result_record(Un,cur_index)
            
            dE = (self.En[cur_index+1] - self.En[cur_index]) / self.tau
            self.tau_list[cur_index+1] = np.max([tau_min, tau_max / np.sqrt(1 + alpha * np.square(np.abs(dE))) ])
            print(f"\r {self.tn[cur_index]:.2f}\\{self.T}, Step size: {self.tau:.2e}, Elapse: {end_time - start_time:.3f} s, "+ msg+" "*10, end="")
            cur_index += 1

        print("")

        self.tn = self.tn[:cur_index+1]
        self.tau_list = self.tau_list[:cur_index+1]
        self.Un = self.Un[:cur_index+1]
        self.En = self.En[:cur_index+1]
        self.Mn = self.Mn[:cur_index+1]
        self.time_elapse = self.time_elapse[:cur_index+1]