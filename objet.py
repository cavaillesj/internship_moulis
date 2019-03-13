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
exec(open("simple_function.py").read(), globals())
import copy


    

class Ode:
    def __init__ (self, model = "allee_effect_adi", Init = None, Param_phy = None, solveur = "euler_ex", Param_num = None, finalTime = None, dt = None, law_amplitude = "exponential", law_freq = "bernoulli", Fire = None):
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
                self.param1 = .4
                self.param2 = .4
                
# =============================================================================
        
        if(Fire is not None):
            self.Fire = Fire
#            print("changement de fire", cf.f_lineno)
        else:
            Param_freq = {"p":0.01}
            Param_ampl = {"scale":0.07}
            
            self.Fire = {"frequence": "bernoulli",
                         "param_freq" : Param_freq,
                         "amplitude": "exponential",
                         "param_amplitude" : Param_ampl,
                         "type" : "proportionnal",
                         "coef_W_N" : 5}
        
#        self.law_freq = law_freq
#        self.law_amplitude = law_amplitude
#        self.coef_W_N = 5.
        
# =============================================================================

# =============================================================================
#       TO DO : automatise the print and the use fr perturbation of this
# =============================================================================
#        Param_freq = {"p":0.01}
#        Param_ampl = {"scale":0.07}
#        
#        self.Fire = {"frequence": law_freq,
#                "param_freq" : Param_freq,
#                "amplitude": law_amplitude,
#                "param_amplitude" : Param_ampl}
        
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


    def solve(self, Init=None, Time=None, solveur = None):
#        if(solveur != None): # allow to change to solveur when we call solve
#            self.solveur = solveur
        
        if(Init is None):
            Init = self.Init
        if(Time is None):
            Time = self.Time
        if(solveur is None):
            solveur = self.solveur
        
        dic_model = {"allee_effect" : self.F_allee_effect, 
                     "allee_effect_adi" : self.F_allee_effect_adi,
                     "verhulst" : self.F_verhulst}
#        dic_solveur = {"odeint" : odeint,
#                       "euler_ex" : self.euler_ex}

        if(solveur == "odeint"):
            Y = odeint(dic_model[self.model], Init, Time)
        elif(solveur == "euler_ex"):
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


    def plot_phase_portrait_2(self, Xwindow = np.array([0,10]), Ywindow = np.array([0,10]), name = "Phase portrait", B_legend = True, show = False):
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
            
#            plt.streamplot(XX, YY, U, V, color = FN)
            plt.streamplot(XX, YY, U, V)

            plt.contourf(XX, YY, FN)

            if(B_legend):
                plt.legend()
        plt.title(name, fontsize=20)
        plt.xlabel("N", fontsize=12)
        plt.ylabel("W", fontsize=12)   
        if(show):
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
#         frequence fire
#        print(self.law_freq, self.law_amplitude)
        
# =============================================================================         
        law_freq = self.Fire["frequence"]
        self.law_amplitude = self.Fire["amplitude"]
        self.coef_W_N = self.Fire["coef_W_N"]
# ============================================================================= 

        if(law_freq == "bernoulli"):
            p = self.Fire["param_freq"]["p"]
            Freq_fire = np.random.binomial(n=1, p=self.dt*p, size = self.NbreIte) # we have to multiply by dt in order to have a perturbation independant to the numerical step
        else:
            print("The law of the fire frequence is not known")

        # amplitude fire
        if(self.law_amplitude == "exponential"):
            Ampl_fire = - np.random.exponential(**self.Fire["param_amplitude"], size = self.NbreIte) 
            self.Perturbation =  np.array(2*[Freq_fire * Ampl_fire])            
            self.Perturbation[1,:] = self.coef_W_N * self.Perturbation[1,:]
#            print(Freq_fire)
        elif(self.law_amplitude == "gamma"):
            Ampl_fire = - np.random.gamma(**self.Fire["param_amplitude"], size = self.NbreIte)
            self.Perturbation =  np.array(2*[Freq_fire * Ampl_fire])
            self.Perturbation[1,:] = self.coef_W_N * self.Perturbation[1,:]
        elif(self.law_amplitude == "lognormal"):
            Ampl_fire = - np.random.lognormal(**self.Fire["param_amplitude"], size = self.NbreIte)
            self.Perturbation =  np.array(2*[Freq_fire * Ampl_fire])
            self.Perturbation[1,:] = self.coef_W_N * self.Perturbation[1,:]
        elif(self.law_amplitude == "power"):
            Ampl_fire = - np.random.power(**self.Fire["param_amplitude"], size = self.NbreIte)
            self.Perturbation =  np.array(2*[Freq_fire * Ampl_fire])
            self.Perturbation[1,:] = self.coef_W_N * self.Perturbation[1,:]
        elif(self.law_amplitude == "multivariate_normal"):
#            mean = np.array([0, 0])
#            cov = np.array([[0.2, 0.2],
#                            [0.2, 0.5]])
            Ampl_fire = - abs(np.random.multivariate_normal(**self.Fire["param_amplitude"]))
            self.Perturbation =  Freq_fire * Ampl_fire        
        else:
            print("The law of the fire amplitude is not known")  
        self.FireB = abs(self.Perturbation[0,:]) > 1e-5
        return self.Perturbation            

        
# =============================================================================
# =============================================================================


Param_freq = {"p":0.01}
Param_ampl = {"scale":0.07}

Fire = {"frequence": "bernoulli",
        "param_freq" : Param_freq,
        "amplitude": "exponential",
        "param_amplitude" : Param_ampl,
        "type" : "proportionnal",
        "coef_W_N" : 5}


O = Ode(model = "allee_effect_adi", Init=[0.5, 0.5], Param_phy= [0.45, 0.45], finalTime = 500, Fire = Fire)
##O.perturbation("neg_poisson", param=[0.2, 0.1])
O.perturbation()
O.solve_by_part()
plt.figure(figsize = (16, 8))
O.plot_time_series()

#O.plot_phase_portrait_2(Xwindow = [0, 1.5], Ywindow = [0, .75])
