#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:10:43 2019

@author: azer
"""

import numpy as np
import matplotlib.pyplot as plt


n0 = 0.1
r = 1.1
t_final = 10


def evolution(n0 = n0, r=r):
    n = n0
    N = [n]
    for i in range(t_final):
        n = r*n*(1-n)
        N += [n] 
    return N

def aff_evolution(N):
    plt.plot(N)
    plt.xlabel("time")
    plt.ylabel("population density")
    plt.title("Evolution of populaiton")
    plt.show()
    return


#N = evolution()
#aff_evolution(N)


def dynamic_r():
    R = np.linspace(0, 4, 100)
    D = np.zeros_like(R)
    for i in range(len(R)):
        N = evolution(r=R[i])
        D[i] = N[-1]
    return D

"""
D = dynamic_r()
plt.plot(D)
plt.xlabel("R")
plt.ylabel("final point")
plt.show()
"""

def logistic_recursive(n, t):
    if(t==0):
        return n
    else:
        return logistic_recursive(r*n*(n-1), t-1) 
    
#print("final point", logistic_recursive(0.1, 10))



# =============================================================================
#   Continuous case
# =============================================================================



def euler(dt, r = r, n0 =n0):
    n = n0
    N = []
#    dt = 0.01
    for i in range(int(t_final//dt)):
        n += dt*r*n*(1-n)
        N +=[n]
    return N

#Dt = [10**-i for i in range(1, 6)]
#
#for i, dt in enumerate(Dt):
#    T = np.arange(0, t_final-dt, dt)
#    N = euler(dt)
#    plt.plot(T, N, label = "dt="+str(dt))
#    plt.xlabel("time")
#    plt.ylabel("N density")
#plt.legend()
#plt.show()


# =============================================================================
#   Adaptive euler
# =============================================================================


def euler_adaptative():
    dt = 0.01 # initial time step
    t = 0
    n = n0
    N = [n]
    T = [t]
    while(t < t_final):########### condition on the approximation
        f = r*n*(1-n)
        if(f>0.2*n):
            dt = dt/2
        elif(f<0.1*n):
            dt = dt*2
        n += dt*f
        t += dt
        N +=[n]
        T +=[t]
    return T, N

T, N = euler_adaptative()
plt.plot(T, N)
plt.show()
