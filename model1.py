#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:02:31 2019
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from plotdf import plotdf



def phy_param():
    g = 1.
    K = 1.
    A = 1.
    m = 1.
    d = 1.
    Param_phy = [g, K, A, m, d]
    return Param_phy


def euler_ex(Init, Param_phy, Param_num):
    N0, W0 = Init
    g, K, A, m, d = Param_phy
    T, dt = Param_num
    N = np.zeros(T)
    W = np.zeros(T)
    N[0] = N0
    W[0] = W0
    for i in range(1, T):
        n = N[i-1]
        w = W[i-1]
        Nder = g*n*(1-n/K)*(n/A-1)
        Wder = m*n - d*w
        N[i] = n + dt*Nder
        W[i] = w + dt*Wder
    return N, W





def F(Y, t=0):
    n, w = Y
    Nder = g*n*(1-n/K)*(n/A-1)
    Wder = m*n - d*w    
    return [Nder, Wder]




def solveur(Init, Param_phy, Param_num, method = "odeint"):
    N, W = Init
    if(method == "euler_ex"):
        N, W = euler_ex(Init, Param_phy, Param_num)    
    elif(method == "odeint"):
        X = np.arange(T)
        Y = odeint(F, Init, X)
        N, W = np.array(Y).transpose()
    else:
        print("This numerical method in not knowed")
    return N, W


Init = [0.5, 0.5]
Param_phy = phy_param()
g, K, A, m, d = Param_phy
T = 10
dt = 0.01
Param_num = [T, dt]
N, W = solveur(Init, Param_phy, Param_num)


X = np.arange(T)
plt.plot(X, N, label="N")
plt.plot(X, W, label="W")
plt.legend()
plt.show()

# =============================================================================
#   Phase portrait
# =============================================================================

plotdf(F,np.array([0,50]), np.array([0,50]))
plt.title("Phase portrait")
plt.xlabel("N")
plt.ylabel("W")         
plt.show()