#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:44:51 2019

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from plotdf import plotdf



class Ode: 
    def __init__ (self, model = "model_1", Init = None, Param_phy = None, Param_num = None):
        self.model = model
        if(Init != None):
            self.Init = Init
        else:
            self.Init = [0.5, 0.5]
        
        if(Param_phy != None):
            self.g, self.K, self.A, self.m, self.d = Param_phy
        else:
            self.g = 1.
            self.K = 1.
            self.A = 1.
            self.m = 1.
            self.d = 1.
        
        if(Param_num != None):
            self.T, self.dt = Param_num
        else:
            self.finalTime = 10
            self.dt = 0.1
        self.NbreIte = int(self.finalTime / self.dt)
        self.Time = np.arange(self.finalTime)

        self.perturbation()
        return

    def F_model_1(self, Y, t):
        """ F for model 1"""
        n, w = Y
        Nder = self.g*n*(1-n/self.K)*(n/self.A-1) + self.Perturbation[int(t/self.dt), 0]
        Wder = self.m*n - self.d*w + self.Perturbation[int(t/self.dt), 1]
        return [Nder, Wder]
        
    def solve(self):
        if(self.model == "model_1"):
            Y = odeint(self.F_model_1, self.Init, self.Time)
        self.N, self.W = np.array(Y).transpose()
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
        plt.title("Time series, \n with perturbation : "+self.law)#+", with parameters : "+str(self.Param_pertubation))
        plt.show()
        
    def plot_phase_portrait(self):
        plotdf(self.F_model_1, np.array([0,10]), np.array([0,10]), parameters={'t':0})
        plt.title("Phase portrait")
        plt.xlabel("N")
        plt.ylabel("W")         
        plt.show()
        
    def perturbation(self, law = "not", param=0):
        """array wit the parturbation"""
        print(law)
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
            
                
        
    
O = Ode(model = "model_1")
O.perturbation("neg_poisson", param=[0.1, 0.1])
O.solve()
O.plot_time_series()
#O.plot_phase_portrait()




