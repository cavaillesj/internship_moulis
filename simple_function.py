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
        return "Fire"+"\nfrequence "+Fire["frequence"]+" "+str(Fire["param_freq"])+"\namplitude "+Fire["amplitude"]+" "+str(Fire["Param_strength"])
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



# =============================================================================
#   compute measure separately
# =============================================================================
    
def collapse(Y):
    eps = 1e-3
    N, W = Y
    if(N[-1] < eps):
        return True
    else:
        return False
    
# =============================================================================
# def collapse(Y):
#     eps = 1e-3
#     N, W = Y
#     if(N[-1] < eps):
#         return True
#     else:
#         return False
# =============================================================================


def variability_always(Y):
    """one among different way to compute the variability"""
    N, W = Y
    return np.var([N, W])



def variability_until(Y):
    """return both variability (compute until it collapse) and collapse"""
    N, W = Y
    eps = 1e-3 # seuil
    time_extinction = np.argmax(N < eps)
    if(N[-1] < eps):
        v = np.var([N[:time_extinction], W[:time_extinction]])
    else:
        v = np.var([N, W])
    return v



def variability_only(Y):
    """return both variability (compute only if it not collapse) and collapse"""
    eps = 1e-3
    N, W = Y
    if(N[-1] < eps):
        v = np.NaN
    else:
        v = np.var([N, W])
    return v


def variability_10(Y):
    """return both variability (just for the 10% interval of the time study) and collapse"""
    N, W = Y
    eps = 1e-3 # seuil
    if(N[-1] < eps):
        v = np.var([N[:len(N)//10], W[:len(N)//10]])
    else:
        v = np.var([N, W])
    return v


def all_measure(Number_of_simulation = 100, mean = True, **kwargs):
    """solve the system several times with different perturbation in order to make an average"""
    ##### Number_of_simulation : between 10 and 1000 usually 100
    Collapse = np.zeros(Number_of_simulation)
    Variability_always = np.zeros_like(Collapse)
    Variability_until = np.zeros_like(Collapse)
    Variability_only = np.zeros_like(Collapse)
    Variability_10 = np.zeros_like(Collapse)
    Collapse_10_b = np.zeros_like(Collapse)
    Collapse_10_m = np.zeros_like(Collapse)

    for i in range(Number_of_simulation):
        O = Ode(**kwargs)
        Y = O.solve_by_part()
        # measures
        Collapse[i] = collapse(Y)
        Variability_always[i] = variability(Y)
        Variability_until[i] = variability_until(Y)
        Variability_only[i] = variability_only(Y)
        Variability_10[i] = variability_10(Y)
        Collapse_10_b[i] = collapse(Y[:,:len(Y[0])//10])
        Collapse_10_m[i] = Y[0,len(Y[0])//10]
    if(mean):
        return np.nanmean(Collapse), np.nanmean(Variability_always), np.nanmean(Variability_until), np.nanmean(Variability_only), np.nanmean(Variability_10), np.nanmean(Collapse_10_b), np.nanmean(Collapse_10_m)
    else:
        return Collapse, Variability_always, Variability_until, Variability_only, Variability_10, Collapse_10_b, Collapse_10_m