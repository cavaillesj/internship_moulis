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

from inspect import currentframe, getframeinfo
cf = currentframe()
#print("python says line ", cf.f_lineno)

import time as tm


class Ode: 
    def __init__ (self, model = "allee_effect", Init = None, Param_phy = None, solveur = "euler_ex", Param_num = None, finalTime = None, dt = None, law_amplitude = "exponential", law_freq = "bernoulli"):
        self.model = model
        if(Init != None):
            self.Init = Init
        else:
            self.Init = [0.5, 0.5]
        
        
        self.solveur = solveur
        finalTime_default = 50
        dt_default = 0.1
        if(Param_num != None):
            self.T, self.dt = Param_num
        elif(finalTime != None and dt == None):
            self.finalTime = finalTime
            self.dt = dt_default
        elif(finalTime == None and dt != None):
            self.dt = dt
            self.finalTime= finalTime_default
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
# =============================================================================

        Param_freq = {"p":0.01}
        Param_ampl = {"scale":0.7}
        
        self.Fire = {"frequence": law_freq,
                "param_freq" : Param_freq,
                "amplitude": law_amplitude,
                "param_amplitude" : Param_ampl}

        self.Perturbation = self.perturbation()
        return

    def F_allee_effect(self, Y, t):
        """ F for model 1"""
        n, w = Y
        Nder = self.g*n*(1-n/self.K)*(n/self.A-1) + self.Perturbation[0, int(t/self.dt)]
        Wder = self.m*n - self.d*w + self.Perturbation[1, int(t/self.dt)]
        return [Nder, Wder]
    

    def F_allee_effect_adi(self, Y, t):
# =============================================================================
#         REMMETRE LES PERTURBATIONS
# =============================================================================
        """ F for allee effect without dimension"""
        #print("t=", t)
        n, w = Y
#        print("\ndt", self.dt)
#        print("t", t)
        if(int(t/self.dt) >= len(self.Perturbation)):
            print("Problem for solving with perturbation (~line 80)")
            Nder = n*(1-n)*(n-self.param1)
            Wder = self.param2*n - w
        else:           
            Nder = n*(1-n)*(n-self.param1) + self.Perturbation[0, int(t/self.dt)]
            Wder = self.param2*n - w + self.Perturbation[1, int(t/self.dt)]
        return [Nder, Wder]

#    def F_model_1(self, Y, t):
#        """ F for model 1"""
#        n, w = Y
#        Nder = self.g*n*(1-n/self.K)*(n/self.A-1) + self.Perturbation[int(t/self.dt), 0]
#        Wder = self.m*n - self.d*w + self.Perturbation[int(t/self.dt), 1]
#        return [Nder, Wder]
    
    def F_verhulst(self, Y, t):
        """ verhulst """
        n, w = Y
        Nder = self.g*n*(1-n/self.K) + self.Perturbation[0, int(t/self.dt)]
        Wder = self.m*n - self.d*w + self.Perturbation[1, int(t/self.dt)]
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


    def solve(self, solveur = None):
        if(solveur != None):
            self.solveur = solveur
        dic_model = {"allee_effect" : self.F_allee_effect, 
                     "allee_effect_adi" : self.F_allee_effect_adi,
                     "verhulst" : self.F_verhulst}
#        dic_solveur = {"odeint" : odeint,
#                       "euler_ex" : self.euler_ex}
        if(self.solveur == "odeint"):
            Y = odeint(dic_model[self.model], self.Init, self.Time)
        elif(self.solveur == "euler_ex"):
            Y = self.euler_ex(dic_model[self.model], self.Init, self.Time)
        else:
            print("(line ", cf.f_lineno, ") The choice of the solveur is not correct")
#        ### check if the density remain positive
# =============================================================================
#         
# =============================================================================
        self.N, self.W = np.array(Y).transpose()
        return self.N, self.W
    
    def plot_time_series(self):
        
      #  print("\n\nlen(self.perturbation", len(self.Perturbation))
       # print("len(self.N)", len(self.N))
        plt.plot(self.Time, self.N, color = "g", label="N")
        plt.plot(self.Time, self.W, color = "maroon", label="W")
#        plt.plot(self.Time[abs(O.Perturbation) > 1e-4], self.N[abs(O.Perturbation) > 1e-4], "*r", label = "Fire\nfrequence "+self.law_freq+" ("+str(0)+")\namplitude "+self.law_amplitude+"("+str(0)+")")
        plt.plot(self.Time[abs(O.Perturbation[0,:]) > 1e-4], self.N[abs(O.Perturbation[0,:]) > 1e-4], "*r", label = "Fire"+"\nfrequence "+self.Fire["frequence"]+" "+str(self.Fire["param_freq"])+"\namplitude "+self.Fire["amplitude"]+" "+str(self.Fire["param_amplitude"]))
        plt.plot(self.Time[abs(O.Perturbation[0,:]) > 1e-4], self.W[abs(O.Perturbation[0,:]) > 1e-4], "*r")
        plt.legend()
        plt.xlabel("time")
        mmax = max([np.max(self.N), np.max(self.W)])
        plt.ylim(0, 1.1*mmax)
        plt.ylabel("density population")
#        plt.title("Time series, \n with perturbation : "+self.law+", with parameters : "+str(self.Param_pertubation))
        if(self.model == "allee_effect_adi"):
            plt.title("Time series \nparam1 = "+str(self.param1)+", param2 = "+str(self.param2))#+", with parameters : "+str(self.Param_pertubation))
        else:
            plt.title("Time series")#+", with parameters : "+str(self.Param_pertubation))
        plt.show()
        
    def plot_phase_portrait(self, Xwindow = np.array([0,10]), Ywindow = np.array([0,10]), name = "Phase portrait"):
        if(self.model == "allee_effect"):
            plotdf(self.F_allee_effect, Xwindow, Ywindow, parameters={'t':0})
        elif(self.model == "allee_effect_adi"):
            plotdf(self.F_allee_effect_adi, Xwindow, Ywindow, parameters={'t':0})
            plt.plot([0, 1, self.param1], [0, self.param2, self.param1*self.param2], "*", label="equilibrium")
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
        if(self.law_freq == "bernoulli"):
            Freq_fire = np.random.binomial(1, 0.01, size = self.NbreIte)
        else:
            print("The law of the fire frequence is not known")
        
        # amplitude fire
        if(self.law_amplitude == "exponential"):
            Ampl_fire = - np.random.exponential(scale = 0.7, size = self.NbreIte)
            self.Perturbation =  np.array(2*[Freq_fire * Ampl_fire])
        elif(self.law_amplitude == "gamma"):
            Ampl_fire = - np.random.gamma(shape = 0.5, scale= 1, size = self.NbreIte)
            self.Perturbation =  np.array(2*[Freq_fire * Ampl_fire])
        elif(self.law_amplitude == "lognormal"):
            Ampl_fire = - np.random.lognormal(mean = -2, sigma=2, size = self.NbreIte)
            self.Perturbation =  np.array(2*[Freq_fire * Ampl_fire])
        elif(self.law_amplitude == "power"):
            Ampl_fire = - np.random.power(a = 1, size = self.NbreIte)
            self.Perturbation =  np.array(2*[Freq_fire * Ampl_fire])
        elif(self.law_amplitude == "multivariate_normal"):
            mean = np.array([0, 0])
            cov = np.array([[0.2, 0.2],
                            [0.2, 0.5]])
            Ampl_fire = - abs(np.random.multivariate_normal(mean, cov))
            self.Perturbation =  Freq_fire * Ampl_fire        
        else:
            print("The law of the fire amplitude is not known")        
        return self.Perturbation            

        
# =============================================================================

O = Ode(model = "allee_effect_adi", Init=[0.5, 0.5], Param_phy= [0.45, 0.45], finalTime = 100)
#O.perturbation("neg_poisson", param=[0.2, 0.1])
O.solve("euler_ex")
O.plot_time_series()
#O.plot_phase_portrait(Xwindow = [0, 2], Ywindow = [0, 2])


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
plt.xlabel("param1")        
plt.ylabel("param2")        
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
plt.ylabel("param1")
ax.plot_wireframe(X, Y, NN_T[:,len(Param2)//4,:], rstride=1, cstride=0)

ax = fig.add_subplot(2, 2, 2, projection='3d')
X, Y = np.meshgrid(O.Time, Param2)
plt.title("N density")  
plt.xlabel("time")
plt.ylabel("param2")
ax.plot_wireframe(X, Y, NN_T[len(Param1)//4,:,:], rstride=1, cstride=0)

ax = fig.add_subplot(2, 2, 3, projection='3d')
X, Y = np.meshgrid(O.Time, Param1)
plt.title("W density")
plt.xlabel("time")
plt.ylabel("param1")
ax.plot_wireframe(X, Y, WW_T[:,len(Param2)//4,:], rstride=1, cstride=0)

ax = fig.add_subplot(2, 2, 4, projection='3d')
X, Y = np.meshgrid(O.Time, Param2)
plt.title("W density")
plt.xlabel("time")
plt.ylabel("param2")
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

for i, param2 in enumerate(Param1):
    for j, param1 in enumerate(Param2):
        plt.subplot(len(Param1), len(Param2), j+1 + len(Param2)*(i))
        O = Ode(model = "allee_effect_adi", Init = Init, Param_phy=[param1, param2])        
        O.plot_phase_portrait(Xwindow = [0, 2], Ywindow = [0, 2], name="param1 = "+str(param1)+", param2 = "+str(param2))

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
plt.ylabel("param1")

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
#plt.ylabel("param1")
#
#Color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']#, 'w']
#for l in range(Number_of_simulation):
#    for i, param1 in enumerate(Param1):
#        ax.plot_wireframe(X[:,i], Y[:,i], NN_T[l,i,:], rstride=1, cstride=0, color = Color[i%len(Color)])
