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


class Ode: 
    def __init__ (self, model = "allee_effect", Init = None, Param_phy = None, Param_num = None, finalTime = None, dt = None):
        self.model = model
        if(Init != None):
            self.Init = Init
        else:
            self.Init = [0.5, 0.5]
        
    
        finalTime_default = 20
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
        self.Time = np.arange(self.finalTime)
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
                
        self.Perturbation = self.perturbation()
        return

    def F_allee_effect(self, Y, t):
        """ F for model 1"""
        n, w = Y
        Nder = self.g*n*(1-n/self.K)*(n/self.A-1) + self.Perturbation[int(t/self.dt), 0]
        Wder = self.m*n - self.d*w + self.Perturbation[int(t/self.dt), 1]
        return [Nder, Wder]

    def F_allee_effect_adi(self, Y, t):
# =============================================================================
#         REMMETRE LES PERTURBATIONS
# =============================================================================
        """ F for model 1"""
        n, w = Y
#        print("\ndt", self.dt)
#        print("t", t)
        Nder = n*(1-n)*(n-self.param1) + self.Perturbation[int(t/self.dt), 0]
        Wder = self.param2*n - w + self.Perturbation[int(t/self.dt), 1]
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
        Nder = self.g*n*(1-n/self.K) + self.Perturbation[int(t/self.dt), 0]
        Wder = self.m*n - self.d*w + self.Perturbation[int(t/self.dt), 1]
        return [Nder, Wder]
        
        
    def solve(self):
        if(self.model == "allee_effect"):
            Y = odeint(self.F_allee_effect, self.Init, self.Time)
        elif(self.model == "allee_effect_adi"):
#            print("self.Init", self.Init)
#            print("self.Time", self.Time)
            Y = odeint(self.F_allee_effect_adi, self.Init, self.Time)
        elif(self.model == "verhulst"):
            Y = odeint(self.F_verhulst, self.Init, self.Time)
        else:
            print("The choie of the model is not correct")
        self.N, self.W = np.array(Y).transpose()
        
        ### check if the density remain positive
        
        return self.N, self.W
    
    def plot_time_series(self):
        plt.plot(self.Time, self.N, color = "r", label="N")
        plt.plot(self.Time, self.W, color = "b", label="W")
        plt.plot(self.Time, self.N+self.Perturbation[:self.finalTime, 0], "*m", label="N pertubation")
        plt.plot(self.Time, self.W+self.Perturbation[:self.finalTime, 1], "*c", label="W pertubation")
        plt.legend()
        plt.xlabel("time")
        plt.ylabel("density population")
#        plt.title("Time series, \n with perturbation : "+self.law+", with parameters : "+str(self.Param_pertubation))
        if(self.model == "allee_effect_adi"):
            plt.title("Time series, \n with perturbation : "+self.law+"\nparam1 = "+str(self.param1)+", param2 = "+str(self.param2))#+", with parameters : "+str(self.Param_pertubation))
        else:
            plt.title("Time series, \n with perturbation : "+self.law)#+", with parameters : "+str(self.Param_pertubation))
        plt.show()
        
    def plot_phase_portrait(self, Xwindow = np.array([0,10]), Ywindow = np.array([0,10]), name = "Phase portrait"):
        if(self.model == "allee_effect"):
            plotdf(self.F_allee_effect, Xwindow, Ywindow, parameters={'t':0})
        elif(self.model == "allee_effect_adi"):
            plotdf(self.F_allee_effect_adi, Xwindow, Ywindow, parameters={'t':0})
        plt.title(name)
        plt.xlabel("N")
        plt.ylabel("W")         
        plt.show()
        
    def perturbation(self, law = "not", param=0):
        """array wit the parturbation"""
        self.law = law
        if(law == "not"):
            self.Perturbation = np.zeros((self.NbreIte, 2))
        elif(law == "poisson"):
    #        if(type(param) != int and type(param) != float):
            if(len(param) != 2):
                print("Error in the parameter choice")
            else:
                lambd, scale = param
                self.Perturbation = scale*np.random.poisson(lambd, [self.NbreIte, 2])
        elif(law == "neg_poisson"):
    #        if(type(param) != int and type(param) != float):
            if(len(param) != 2):
                print("Error in the parameter choice")
            else:
                lambd, scale = param
                self.Perturbation = -abs(scale*np.random.poisson(lambd, [self.NbreIte, 2]))
        elif(law == "gaussian"):
            if(len(param) != 2):
                print("Error in the parameter choice")
            else:
                self.Perturbation =  np.random.normal(param[0], param[1], [self.NbreIte, 2])
        elif(law == "neg_gaussian"):
            if(len(param) != 2):
                print("Error in the parameter choice")
            else:
                self.Perturbation =  -abs(np.random.normal(param[0], param[1], [self.NbreIte, 2]))        
        else:
            print("the choice of the perturbation is not correct")
        return self.Perturbation
            
                
# =============================================================================
    
 
O = Ode(model = "allee_effect_adi", Init=[0.5, 0.5], Param_phy= [0.45, 0.45], finalTime = 100)
O.perturbation("neg_poisson", param=[0.2, 0.1])
O.solve()
O.plot_time_series()
#O.plot_phase_portrait(Xwindow = [0, 2], Ywindow = [0, 2])
# 
# =============================================================================

# =============================================================================
#   Final point for different value of param1 and param2
# =============================================================================
"""
Param1 = np.linspace(0, 2, 21)
Param2 = np.linspace(0, 2, 21)
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

NN_T = np.zeros((len(Param1), len(Param2), O.finalTime))
WW_T = np.zeros_like(NN_T)

Init = [0.5, 0.5]
for i, param1 in enumerate(Param1):
    for j, param2 in enumerate(Param2):
        O = Ode(model = "allee_effect_adi", Init = Init, Param_phy=[param1, param2])
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
ax.plot_wireframe(X, Y, NN_T[:,len(Param2)//4,:], rstride=2, cstride=2)

ax = fig.add_subplot(2, 2, 2, projection='3d')
X, Y = np.meshgrid(O.Time, Param2)
plt.title("N density")  
plt.xlabel("time")
plt.ylabel("param2")
ax.plot_wireframe(X, Y, NN_T[len(Param1)//4,:,:], rstride=2, cstride=2)

ax = fig.add_subplot(2, 2, 3, projection='3d')
X, Y = np.meshgrid(O.Time, Param1)
plt.title("W density")
plt.xlabel("time")
plt.ylabel("param1")
ax.plot_wireframe(X, Y, WW_T[:,len(Param2)//4,:], rstride=2, cstride=2)

ax = fig.add_subplot(2, 2, 4, projection='3d')
X, Y = np.meshgrid(O.Time, Param2)
plt.title("W density")
plt.xlabel("time")
plt.ylabel("param2")
ax.plot_wireframe(X, Y, WW_T[len(Param1)//4,:,:], rstride=2, cstride=2)
plt.show()





# =============================================================================
# PHASE PORTRAIT FOR DIFFERENTS VALUES OF PARAM 1 & 2
# =============================================================================

plt.figure(figsize = (16, 16))
plt.title("Phase portrait")
Param1 = np.linspace(0, 1.5, 4)
Param2 = np.linspace(0, 1.5, 4)

for i, param2 in enumerate(Param1):
    for j, param1 in enumerate(Param2):
        plt.subplot(len(Param1), len(Param2), j+1 + len(Param2)*(i))
        O = Ode(model = "allee_effect_adi", Init = Init, Param_phy=[param1, param2])        
        O.plot_phase_portrait(Xwindow = [0, 2], Ywindow = [0, 2], name="param1 = "+str(param1)+", param2 = "+str(param2))

"""
