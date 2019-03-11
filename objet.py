#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:44:51 2019

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from plotdf import plotdf

from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm
#from matplotlib import mpl

from inspect import currentframe, getframeinfo
cf = currentframe()
#print("python says line ", cf.f_lineno)

import time as tm

import copy


def seuil(P):
    for i,p in enumerate(P):
        if(p > 1):
            P[i] = 1
    return P            
    
def Fire_print(Fire, coef_W_N = None):
    """return for the dictionnary Fire the different parameters use for the fire in order to make a complete print"""
    if(coef_W_N == None):
        return "Fire"+"\nfrequence "+Fire["frequence"]+" "+str(Fire["param_freq"])+"\namplitude "+Fire["amplitude"]+" "+str(Fire["param_amplitude"])
    else:
        return "Fire proportional to N and "+str(coef_W_N)+"W\nfrequence "+Fire["frequence"]+" "+str(Fire["param_freq"])+"\namplitude "+Fire["amplitude"]+" "+str(Fire["param_amplitude"])


class Ode:
    def __init__ (self, model = "allee_effect", Init = None, Param_phy = None, solveur = "euler_ex", Param_num = None, finalTime = None, dt = None, law_amplitude = "exponential", law_freq = "bernoulli"):
        self.model = model
        if(Init != None):
            self.Init = Init
        else:
            self.Init = [0.5, 0.5]
        
        
        self.solveur = solveur
        finalTime_default = 500
        dt_default = 1.0
        if(Param_num != None):
            self.T, self.dt = Param_num
        elif(finalTime != None and dt == None):
            self.finalTime = finalTime
            self.dt = dt_default
        elif(finalTime == None and dt != None):
            self.dt = dt
            self.finalTime= finalTime_default
        elif(finalTime != None and dt != None):
            self.dt = dt
            self.finalTime= finalTime
        else:
            self.finalTime = finalTime_default
            self.dt = dt_default
        self.NbreIte = int(self.finalTime / self.dt)
        self.Time = np.arange(0, self.finalTime, self.dt)
#        self.Time = np.arange(self.finalTime)
#        self.perturbation()

        if(model == "allee_effect"):
            if(Param_phy != None):
                self.g, self.K, self.A, self.m, self.d = Param_phy
            else:
                self.g = 1.
                self.K = 1.
                self.A = 1.
                self.m = 1.
                self.d = 1.
        elif(model == "allee_effect_adi"):
            if(Param_phy != None):
                self.param1, self.param2 = Param_phy
            else:
                self.param1 = 1.
                self.param2 = 1.
                
# =============================================================================
        self.law_freq = law_freq
        self.law_amplitude = law_amplitude
        self.coef_W_N = 5.
# =============================================================================

# =============================================================================
#       TO DO : automatise the print and the use fr perturbation of this
# =============================================================================
        Param_freq = {"p":0.01}
        Param_ampl = {"scale":0.07}
        
        self.Fire = {"frequence": law_freq,
                "param_freq" : Param_freq,
                "amplitude": law_amplitude,
                "param_amplitude" : Param_ampl}

        self.Perturbation = self.perturbation()
        return

    def copy(self):
        return copy.copy(self)        



    def F_allee_effect(self, Y, t):
        """ F for model 1"""
        n, w = Y
        Nder = self.g*n*(1-n/self.K)*(n/self.A-1)
        Wder = self.m*n - self.d*w
        return [Nder, Wder]
    

    def F_allee_effect_adi(self, Y, t):
        """ F for allee effect without dimension"""
        #print("t=", t)
        n, w = Y
#        print("\ndt", self.dt)
#        print("t", t)
        Nder = n*(1-n)*(n-self.param1)
        Wder = self.param2*n - w
        return [Nder, Wder]

    
    def F_verhulst(self, Y, t):
        """ verhulst """
        n, w = Y
        Nder = self.g*n*(1-n/self.K)
        Wder = self.m*n - self.d*w
        return [Nder, Wder]
                

                                                
#    def solve(self):        
#        if(self.solveur == "odeint"):
#            if(self.model == "allee_effect"):
#                Y = odeint(self.F_allee_effect, self.Init, self.Time)
#            elif(self.model == "allee_effect_adi"):
#    #            print("self.Init", self.Init)
#    #            print("self.Time", self.Time)
#                #print(self.Time)
#                Y = odeint(self.F_allee_effect_adi, self.Init, self.Time)
#            elif(self.model == "verhulst"):
#                Y = odeint(self.F_verhulst, self.Init, self.Time)
#            else:
#                print("The choie of the model is not correct")
#        elif(self.solveur == "euler_ex"):
#            pass            
#        else:
#            print("(line ", cf.f_lineno, ") The choice of the solveur is not correct")
#        ### check if the density remain positive
# =============================================================================
#         
# =============================================================================
#        self.N, self.W = np.array(Y).transpose()
#        return self.N, self.W
#    

    def euler_ex(self, F, Init, Time):
        """ compute the solution with explicit euler method"""
        Y = np.zeros((len(Time), len(Init)))
        Y[0,:] = Init
#        print(type(Init))
#        print(type(Y[1, :]))
#        print(type(F[1,:]))        
        for i in range(1, len(Time)):
            Y[i,:] = Y[i-1,:] + self.dt*np.array(F(Y[i-1,:], Time[i])) # t or i ??????
        
        return Y


    def solve(self, Init, Time):
#        if(solveur != None): # allow to change to solveur when we call solve
#            self.solveur = solveur
            
        dic_model = {"allee_effect" : self.F_allee_effect, 
                     "allee_effect_adi" : self.F_allee_effect_adi,
                     "verhulst" : self.F_verhulst}
#        dic_solveur = {"odeint" : odeint,
#                       "euler_ex" : self.euler_ex}

        if(self.solveur == "odeint"):
            Y = odeint(dic_model[self.model], Init, Time)
        elif(self.solveur == "euler_ex"):
            Y = self.euler_ex(dic_model[self.model], Init, Time)
        else:
            print("(line ", cf.f_lineno, ") The choice of the solveur is not correct")
#        ### check if the density remain positive
# =============================================================================
#         
# =============================================================================
#        self.N, self.W = np.array(Y).transpose()
#        return self.N, self.W
        return Y
    
    
    def solve_by_part(self):
        """solve the ode between two fires"""
#        Y = self.solve(self.Init, self.Time)
#        self.N, self.W = np.array(Y).transpose()
#        return self.N, self.W
        Y = np.zeros((self.NbreIte, 2))
        c = 0           # compteur
        Init = self.Init
        while(c < len(self.FireB)):
            if(self.FireB[c] == False):
                c_old = c
                Sequence = [self.Time[c]]
                c += 1               
                while(c < len(self.FireB) and self.FireB[c] == False):
                    Sequence += [self.Time[c]]
                    c+=1
                Y[c_old:c] = self.solve(Init, Sequence) 
#                if(c < len(FireB)): ### they are a fire !!
#                    Y[c] = Y[c-1] #+ self.Perturbation[:,c]
#                    Init = Y[c]
            else:
                # fire !
                Init = Y[c-1] + seuil(self.Perturbation[:,c])*Y[c-1]
#                Y[c] = np.array([np.NAN, np.NAN])
                if(Init[0] < 0):
                    Init[0] = 0
                if(Init[1] < 0):
                    Init[1] = 0
                Y[c] = Init
                c += 1
        Y = np.array(Y).transpose()
        self.N, self.W = Y
        return Y
        
    
    def plot_time_series(self):
        
      #  print("\n\nlen(self.perturbation", len(self.Perturbation))
       # print("len(self.N)", len(self.N))
        plt.plot(self.Time, self.N, color = "g", label="N")
        plt.plot(self.Time, self.W, color = "maroon", label="W")
#        plt.plot(self.Time[abs(O.Perturbation) > 1e-4], self.N[abs(O.Perturbation) > 1e-4], "*r", label = "Fire\nfrequence "+self.law_freq+" ("+str(0)+")\namplitude "+self.law_amplitude+"("+str(0)+")")

        before_fire = list(self.FireB[1:])+[False]
        plt.plot(self.Time[abs(O.Perturbation[0,:]) > 1e-4], self.N[before_fire], "*r", label = Fire_print(self.Fire, coef_W_N = self.coef_W_N))
        plt.plot(self.Time[abs(O.Perturbation[0,:]) > 1e-4], self.W[before_fire], "*r")
        plt.legend()
        plt.xlabel("time")
        mmax = max([max(self.N), max(self.W)])

#        plt.ylim(0, 1.1*mmax)

        plt.ylabel("density")
#        plt.title("Time series, \n with perturbation : "+self.law+", with parameters : "+str(self.Param_pertubation))
        if(self.model == "allee_effect_adi"):
            plt.title("Time series \na = "+str(self.param1)+", m = "+str(self.param2))#+", with parameters : "+str(self.Param_pertubation))
        else:
            plt.title("Time series")#+", with parameters : "+str(self.Param_pertubation))
        plt.show()
        
    def plot_phase_portrait(self, Xwindow = np.array([0,10]), Ywindow = np.array([0,10]), name = "Phase portrait", B_legend = True):
        if(self.model == "allee_effect"):
            plotdf(self.F_allee_effect, Xwindow, Ywindow, parameters={'t':0})
        elif(self.model == "allee_effect_adi"):
            plotdf(self.F_allee_effect_adi, Xwindow, Ywindow, parameters={'t':0})
            if(self.param1 < 1):
                plt.plot([0, 1], [0, self.param2], "o", label="stable\nequilibrium")
                plt.plot([self.param1], [self.param1*self.param2], "*", label="unstable\nequilibrium")
            else:
                plt.plot([0, self.param1], [0, self.param1*self.param2], "o", label="stable\nequilibrium")
                plt.plot([1], [self.param2], "*", label="unstable\nequilibrium")
                
            if(B_legend):
                plt.legend()
        plt.title(name)
        plt.xlabel("N")
        plt.ylabel("W")         
        plt.show()

    def plot_phase_portrait_0(self, Xwindow = np.array([0,10]), Ywindow = np.array([0,10]), name = "Phase portrait", B_legend = True):
        if(self.model == "allee_effect"):
            print("To DO (line ", cf.f_lineno)
        elif(self.model == "allee_effect_adi"):
#            plotdf(self.F_allee_effect_adi, Xwindow, Ywindow, parameters={'t':0})
            if(self.param1 < 1):
                plt.plot([0, 1], [0, self.param2], "o", label="stable\nequilibrium")
                plt.plot([self.param1], [self.param1*self.param2], "*", label="unstable\nequilibrium")
            else:
                plt.plot([0, self.param1], [0, self.param1*self.param2], "o", label="stable\nequilibrium")
                plt.plot([1], [self.param2], "*", label="unstable\nequilibrium")
            
            X = np.linspace(Xwindow[0], Xwindow[1], 10)
            Y = np.linspace(Ywindow[0], Ywindow[1], 10)
            for i, x in enumerate(X):
                for j, y in enumerate(Y):
#                    print(x, y)
                    O2 = O.copy()
                    O2.Perturbation = np.zeros_like(O2.Perturbation)
                    O2.Init = [x, y]
                    N, W = O2.solve_by_part()
                    eps = 1e-2
                    if(N[-1] < eps):
                        plt.plot(N, W, "black")
#                        print(N[1]-N[0], W[1]-W[0])
#                        plt.quiver(N[1]-N[0], W[1]-W[0])#, "black")
                    elif(abs(N[-1] - 1) < eps):
                        plt.plot(N, W, "blue")
                    elif(abs(N[-1] - 1) < eps):
                        plt.plot(N, W, "orange")
                    else:
                        plt.plot(N, W, "magenta")
                        print("No convergence yet ...")
                    plt.plot(color = "black", label= "extinctions")
                    plt.plot(color = "blue", label= "equilibrium state")
                    plt.plot(color = "orange", label= "equilibrium state")                    
            if(B_legend):
                plt.legend()
        plt.title(name)
        plt.xlabel("N")
        plt.ylabel("W")         
        plt.show()


    def plot_phase_portrait_2(self, Xwindow = np.array([0,10]), Ywindow = np.array([0,10]), name = "Phase portrait", B_legend = True):
        if(self.model == "allee_effect"):
            print("To DO (line ", cf.f_lineno)
        elif(self.model == "allee_effect_adi"):
#            plotdf(self.F_allee_effect_adi, Xwindow, Ywindow, parameters={'t':0})
            if(self.param1 < 1):
                plt.plot([0, 1], [0, self.param2], "o", label="stable\nequilibrium")
                plt.plot([self.param1], [self.param1*self.param2], "*", label="unstable\nequilibrium")
            else:
                plt.plot([0, self.param1], [0, self.param1*self.param2], "o", label="stable\nequilibrium")
                plt.plot([1], [self.param2], "*", label="unstable\nequilibrium")
            
            X = np.linspace(Xwindow[0], Xwindow[1], 10)
            Y = np.linspace(Ywindow[0], Ywindow[1], 10)
            FN = np.zeros((len(X), len(Y)))
            FW = np.zeros_like(FN)         
            
            for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    O2 = O.copy()
                    O2.Perturbation = np.zeros_like(O2.Perturbation)
                    O2.Init = [x, y]
                    N, W = O2.solve_by_part()
                    FN[i,j], FW[i,j] = N[-1], W[-1]

            eps = 1e-2
            Extinction = FN < eps
            Equilibrium1 = abs(FN-1) < eps
            Equilibrium2 = abs(FN-self.param1) < eps

#            FN[:,:] = 0
            FN[Extinction] = 0
            FN[Equilibrium1] = 1
            FN[Equilibrium2] = self.param1
            
            
            FN = FN.transpose()

            
            
#            Color = np.array([["b"]*len(X)]*len(Y))
#            Color[Extinction] = "black"
#            Color[Equilibrium1] = "blue"
#            Color[Equilibrium2] = "orange"
            
            
            ### make a condition if either of the above is not verify, print a message warning


#            plt.plot(color = "black", label= "extinctions")
#            plt.plot(color = "blue", label= "equilibrium state")
#            plt.plot(color = "orange", label= "equilibrium state")                    

            XX, YY = np.meshgrid(X, Y)
            U = XX*(1-XX)*(XX-self.param1)
            V = self.param2*XX - YY
            
#            print(Extinction)
            
#            print("\n", np.shape(XX[Extinction]))
#            print(np.shape(U[Extinction]))
            plt.streamplot(XX, YY, U, V, color = FN)
            
            plt.contourf(XX, YY, FN)

            if(B_legend):
                plt.legend()
        plt.title(name)
        plt.xlabel("N")
        plt.ylabel("W")         
        plt.show()


        
#    def perturbation(self, law = "not", param=0):
#        """array wit the parturbation"""
#        self.law = law
#        if(law == "not"):
#            self.Perturbation = np.zeros(self.NbreIte)
#        elif(law == "poisson"):
#    #        if(type(param) != int and type(param) != float):
#            if(len(param) != 2):
#                print("Error in the parameter choice")
#            else:
#                lambd, scale = param
#                self.Perturbation = scale*np.random.poisson(lambd, self.NbreIte)
#        elif(law == "neg_poisson"):
#    #        if(type(param) != int and type(param) != float):
#            if(len(param) != 2):
#                print("Error in the parameter choice")
#            else:
#                lambd, scale = param
#                self.Perturbation = -abs(scale*np.random.poisson(lambd, self.NbreIte))
#        elif(law == "gaussian"):
#            if(len(param) != 2):
#                print("Error in the parameter choice")
#            else:
#                self.Perturbation =  np.random.normal(param[0], param[1], self.NbreIte)
#        elif(law == "neg_gaussian"):
#            if(len(param) != 2):
#                print("Error in the parameter choice")
#            else:
#                self.Perturbation =  -abs(np.random.normal(param[0], param[1], self.NbreIte))        
#        else:
#            print("the choice of the perturbation is not correct")
#        return self.Perturbation

    def perturbation(self):       
        # frequence fire
#        print(self.law_freq, self.law_amplitude)
        if(self.law_freq == "bernoulli"):
            Freq_fire = np.random.binomial(1, self.dt*0.01, size = self.NbreIte)
        else:
            print("The law of the fire frequence is not known")


        # amplitude fire
        if(self.law_amplitude == "exponential"):
            Ampl_fire = - np.random.exponential(scale = 0.07, size = self.NbreIte)
            self.Perturbation =  np.array(2*[Freq_fire * Ampl_fire])
            self.Perturbation[1,:] = self.coef_W_N * self.Perturbation[1,:]
#            print(Freq_fire)
        elif(self.law_amplitude == "gamma"):
            Ampl_fire = - np.random.gamma(shape = 0.5, scale= 1, size = self.NbreIte)
            self.Perturbation =  np.array(2*[Freq_fire * Ampl_fire])
            self.Perturbation[1,:] = self.coef_W_N * self.Perturbation[1,:]
        elif(self.law_amplitude == "lognormal"):
            Ampl_fire = - np.random.lognormal(mean = -2, sigma=2, size = self.NbreIte)
            self.Perturbation =  np.array(2*[Freq_fire * Ampl_fire])
            self.Perturbation[1,:] = self.coef_W_N * self.Perturbation[1,:]
        elif(self.law_amplitude == "power"):
            Ampl_fire = - np.random.power(a = 1, size = self.NbreIte)
            self.Perturbation =  np.array(2*[Freq_fire * Ampl_fire])
            self.Perturbation[1,:] = self.coef_W_N * self.Perturbation[1,:]
        elif(self.law_amplitude == "multivariate_normal"):
            mean = np.array([0, 0])
            cov = np.array([[0.2, 0.2],
                            [0.2, 0.5]])
            Ampl_fire = - abs(np.random.multivariate_normal(mean, cov))
            self.Perturbation =  Freq_fire * Ampl_fire        
        else:
            print("The law of the fire amplitude is not known")  
        self.FireB = abs(self.Perturbation[0,:]) > 1e-5
        return self.Perturbation            

        
# =============================================================================


O = Ode(model = "allee_effect_adi", Init=[0.5, 0.5], Param_phy= [0.45, 0.45], finalTime = 500)
##O.perturbation("neg_poisson", param=[0.2, 0.1])
#O.perturbation()
O.solve_by_part()
#O.plot_time_series()
O.plot_phase_portrait_2(Xwindow = [0, 1.5], Ywindow = [0, .75])


# =============================================================================
#   Time calculation for euler explicit and odeint (python library)
# =============================================================================

"""
#Dt = [0.1, 0.01, 0.001, 0.0001, 0.1**4, 0.1**5, 0.1**6, 0.1**7, 0.1**8]
FinalTime = [10**i for i in range(2, 6)]
Time_calculation = {"time": FinalTime,
                    "odeint": [], 
                    "euler_ex": []}
for i, finalTime in enumerate(FinalTime):
    O = Ode(model = "allee_effect_adi", Init=[0.5, 0.5], Param_phy= [0.45, 0.45], finalTime = finalTime)
    t0 = tm.time()
    O.solve("odeint")
    t1 = tm.time()
    O.solve("euler_ex")
    t2 = tm.time()
    Time_calculation["odeint"] += [t1-t0]
    Time_calculation["euler_ex"] += [t2-t1]    

plt.loglog(Time_calculation["time"], Time_calculation["odeint"], "+-", label="odeint") 
plt.loglog(Time_calculation["time"], Time_calculation["euler_ex"], "+-", label="euler_ex")
plt.legend()
plt.title("Time calculation")
plt.xlabel("Final time (log scale)")
plt.ylabel("Calculation time in secondes (log scale)")
plt.show()
"""

# =============================================================================
#   Final point for different value of param1 and param2
# =============================================================================

"""
Param1 = np.linspace(0.2, 0.6, 5)
Param2 = np.linspace(0.2, 0.6, 5)
NN = np.zeros((len(Param1), len(Param2)))
WW = np.zeros_like(NN)

m = np.max([np.max(NN), np.max(WW)])

Init = [0.5, 0.5]
for i, param1 in enumerate(Param1):
    #print(i, param1)
    for j, param2 in enumerate(Param2):
        O = Ode(model = "allee_effect_adi", Init = Init, Param_phy=[param1, param2])
        Y = O.solve()
        NN[i,j] = Y[0][-1]
        WW[i,j] = Y[1][-1]


#fig, [ax1, ax2] = plt.subplots(figsize = (16, 8), ncols = 2)
fig, ax = plt.subplots(figsize = (16, 8), ncols = 1)
plt.title("N density")
plt.xlabel("a")        
plt.ylabel("m")        
p1 = ax.imshow(NN, vmin = 0, vmax = m)
fig.colorbar(p1, ax = ax)

fig, ax = plt.subplots(figsize = (16, 8), ncols = 1)
plt.title("W density")
plt.xlabel("param1")        
plt.ylabel("param2")        
p2 = ax.imshow(WW, vmin = 0, vmax = m)
fig.colorbar(p2, ax = ax)

# =============================================================================
#   Time series for differents values of param1 and param2
# =============================================================================


Param1 = np.linspace(0.2, 0.6, 5)
Param2 = np.linspace(0.2, 0.6, 5)
finaltime = 50

NN_T = np.zeros((len(Param1), len(Param2), O.NbreIte))
WW_T = np.zeros_like(NN_T)

Init = [0.5, 0.5]
for i, param1 in enumerate(Param1):
    for j, param2 in enumerate(Param2):
        O = Ode(model = "allee_effect_adi", Init = Init, Param_phy=[param1, param2], finalTime=finaltime)
       # print(O.NbreIte)
        #print(O.Time)
        Y = O.solve()
        NN_T[i,j,:] = Y[0]
        WW_T[i,j] = Y[1]

#fig, [ax1, ax2] = plt.subplots(figsize = (16, 16), ncols = 2, nrows=2)

fig = plt.figure(figsize= (16, 16))

ax = fig.add_subplot(2, 2, 1, projection='3d')
X, Y = np.meshgrid(O.Time, Param1)
plt.title("N density")
plt.xlabel("time")
plt.ylabel("a")
ax.plot_wireframe(X, Y, NN_T[:,len(Param2)//4,:], rstride=1, cstride=0)

ax = fig.add_subplot(2, 2, 2, projection='3d')
X, Y = np.meshgrid(O.Time, Param2)
plt.title("N density")  
plt.xlabel("time")
plt.ylabel("m")
ax.plot_wireframe(X, Y, NN_T[len(Param1)//4,:,:], rstride=1, cstride=0)

ax = fig.add_subplot(2, 2, 3, projection='3d')
X, Y = np.meshgrid(O.Time, Param1)
plt.title("W density")
plt.xlabel("time")
plt.ylabel("a")
ax.plot_wireframe(X, Y, WW_T[:,len(Param2)//4,:], rstride=1, cstride=0)

ax = fig.add_subplot(2, 2, 4, projection='3d')
X, Y = np.meshgrid(O.Time, Param2)
plt.title("W density")
plt.xlabel("time")
plt.ylabel("m")
ax.plot_wireframe(X, Y, WW_T[len(Param1)//4,:,:], rstride=1, cstride=0)
plt.show()

"""

# =============================================================================
# PHASE PORTRAIT FOR DIFFERENTS VALUES OF PARAM 1 & 2
# =============================================================================

"""

plt.figure(figsize = (16, 16))
plt.title("Phase portrait")
Param1 = np.linspace(0.4, 0.6, 3)
Param2 = np.linspace(0.4, 0.6, 3)
Init = [0.5, 0.5]

for i, param2 in enumerate(Param1):
    for j, param1 in enumerate(Param2):
        plt.subplot(len(Param1), len(Param2), j+1 + len(Param2)*(i))
        O = Ode(model = "allee_effect_adi", Init = Init, Param_phy=[param1, param2])        
        O.plot_phase_portrait(Xwindow = [0, 1.5], Ywindow = [0, .75], name="param1 = "+str(param1)+", param2 = "+str(param2), B_legend= False)
plt.legend(loc=(1., 3.))    

"""

# =============================================================================
#   times series with perturbation
# =============================================================================

"""
Param1 = np.linspace(0, 1, 11)
Number_of_simulation = 7

NN_T = np.zeros((Number_of_simulation, len(Param1), O.NbreIte))
WW_T = np.zeros_like(NN_T)

Init = [0.5, 0.5]
param2 = 0.5
for l in range(Number_of_simulation):
    for i, param1 in enumerate(Param1):
        O = Ode(model = "allee_effect_adi", Init = Init, Param_phy=[param1, param2], finalTime = 50)
        O.perturbation("neg_poisson", param=[0.2, 0.2])
        Y = O.solve()
        NN_T[l, i,:] = Y[0]
        WW_T[l, i,:] = Y[1]


#fig, [ax1, ax2] = plt.subplots(figsize = (16, 16), ncols = 2, nrows=2)

fig = plt.figure(figsize= (16, 16))

ax = fig.add_subplot(1, 1, 1, projection='3d')
X, Y = np.meshgrid(O.Time, Param1)
plt.title("N density")
plt.xlabel("time")
plt.ylabel("a")

################ color by simulation

Color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']#, 'w']
for l in range(Number_of_simulation):
    ax.plot_wireframe(X, Y, NN_T[l,:,:], rstride=1, cstride=0, color = Color[l])

"""
################## color by param1

#fig = plt.figure(figsize= (16, 16))
#ax = fig.add_subplot(1, 1, 1, projection='3d')
#plt.title("N density")
#plt.xlabel("time")
#plt.ylabel("a")
#
#Color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']#, 'w']
#for l in range(Number_of_simulation):
#    for i, param1 in enumerate(Param1):
#        ax.plot_wireframe(X[:,i], Y[:,i], NN_T[l,i,:], rstride=1, cstride=0, color = Color[i%len(Color)])




# =============================================================================
#   Final point in color in the space of the param 1 & 2
# =============================================================================


"""
Param1 = np.linspace(0.3, 0.6, 10)
Param2 = np.linspace(0.2, 2., 30)
Final_N = np.zeros((len(Param1), len(Param2)))
Final_W = np.zeros_like(Final_N)

for i, param1 in enumerate(Param1):
    for j, param2 in enumerate(Param2):
        O = Ode(model = "allee_effect_adi", solveur = "odeint", Init=[0.5, 0.5], Param_phy= [param1, param2], finalTime = 50)
        O.perturbation()
        Y  = O.solve_by_part()
        Final_N[i,j], Final_W[i,j] = Y[:,-1]



plt.figure(figsize = (16, 16))
#plt.title("Final point for different parameters")
plt.suptitle("Pertubation "+Fire_print(O.Fire))

mmax = max([np.max(Final_N), np.max(Final_W)])
plt.subplot(1,2,1)
extent = (Param1[0], Param1[-1], Param2[0], Param2[-1])
plt.imshow(Final_N[:,::-1].transpose(), extent = extent, vmin = 0, vmax = mmax, aspect = "auto")
#plt.colorbar()
plt.title("W final point")
plt.xlabel("a")
plt.ylabel("m")

plt.subplot(1,2,2)
plt.imshow(Final_W[:,::-1].transpose(), extent = extent, vmin = 0, vmax = mmax, aspect = "auto")
plt.colorbar()
plt.title("N final point")
plt.xlabel("a")
plt.ylabel("m")

"""


