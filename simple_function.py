#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:29:48 2019
"""

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

def variability(Y):
    """one among different way to compute the variability"""
    N, W = Y
    return np.var([N, W])




def collapse(Y):
    eps = 1e-3
    N, W = Y
    if(N[-1] < eps):
        return True
    else:
        return False


def variability_collapse_until(Y):
    """return both variability (compute until it collapse) and collapse"""
    N, W = Y
    eps = 1e-3 # seuil
    time_extinction = np.argmax(N < eps)
    if(N[-1] < eps):
        c = True
        v = np.var([N[:time_extinction], W[:time_extinction]])
    else:
        c = False
        v = np.var([N, W])
    return v, c



def variability_collapse_only(Y):
    """return both variability (compute only if it not collapse) and collapse"""
    eps = 1e-3
    N, W = Y
    if(N[-1] < eps):
        c = True
        v = np.NaN
    else:
        c = False
        v = np.var([N, W])
    return v, c


def variability_collapse_10(Y):
    """return both variability (just for the 10% interval of the time study) and collapse"""
    N, W = Y
    eps = 1e-3 # seuil
    if(N[-1] < eps):
        c = True
        v = np.var([N[:len(N)//10], W[:len(N)//10]])
    else:
        c = False
        v = np.var([N, W])
    return v, c
    
