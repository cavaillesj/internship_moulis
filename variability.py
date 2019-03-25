# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:21:25 2019

@author: jerome
"""

import numpy as np



# =============================================================================
# def variability(Y):
#     """one among different way to compute the variability"""
#     N, W = Y
#     return np.var([N, W])
# 
# def collapse(Y):
#     eps = 1e-3
#     N, W = Y
#     if(N[-1] < eps):
#         return True
#     else:
#         return False
# 
# 
# def variability_collapse_until(Y):
#     """return both variability (compute until it collapse) and collapse"""
#     N, W = Y
#     eps = 1e-3 # seuil
#     time_extinction = np.argmax(N < eps)
#     if(N[-1] < eps):
#         c = True
#         v = np.var([N[:time_extinction], W[:time_extinction]])
#     else:
#         c = False
#         v = np.var([N, W])
#     return v, c
# 
# 
# 
# def variability_collapse_only(Y):
#     """return both variability (compute only if it not collapse) and collapse"""
#     eps = 1e-3
#     N, W = Y
#     if(N[-1] < eps):
#         c = True
#         v = np.NaN
#     else:
#         c = False
#         v = np.var([N, W])
#     return v, c
# 
# 
# def variability_collapse_10(Y):
#     """return both variability (just for the 10% interval of the time study) and collapse"""
#     N, W = Y
#     eps = 1e-3 # seuil
#     if(N[-1] < eps):
#         c = True
#         v = np.var([N[:len(N)//10], W[:len(N)//10]])
#     else:
#         c = False
#         v = np.var([N, W])
#     return v, c
# =============================================================================


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

eps = 1e-3 # perhaps 1e-2 is enough better ...

def variability_always(Y):
    """one among different way to compute the variability"""
    N, W = Y
    return np.var(W)



def variability_until(Y, eps = eps):
    """return both variability (compute until it collapse) and collapse"""
    N, W = Y
    time_extinction = np.argmax(N < eps)
    if(N[-1] < eps):
        v = np.var(W[:time_extinction])
    else:
        v = np.var(W)
    return v



def variability_only(Y, eps = eps):
    """return both variability (compute only if it not collapse) and collapse"""
    N, W = Y
    if(N[-1] < eps):
        v = np.NaN
    else:
        v = np.var(W)
    return v


def variability_10(Y, eps = eps):
    """return both variability (just for the 10% interval of the time study) and collapse"""
    N, W = Y
    if(N[-1] < eps):
        v = np.var([N[:len(N)//10], W[:len(N)//10]])
    else:
        v = np.var(W)
    return v



def variability_half(Y, eps = eps):
    """return both variability (compute until it collapse) and collapse"""
    N, W = Y
     # seuil
    if(N[-1] < eps):    
        time_extinction = np.argmax(N < eps)    
    else: # they are no extinction
        time_extinction = len(N)#//2
    final_time_variability_computation = time_extinction // 2
    average = np.mean(W[:final_time_variability_computation])
    initial_time_variability_computation = np.argmax(W < average)
    if(final_time_variability_computation - initial_time_variability_computation >= 0.1*len(N)): # need enough data to make relevent computation
        return np.var(W[initial_time_variability_computation:final_time_variability_computation])
    else:
        return np.NaN
        
    
    
def speed_collapse(Y, eps = eps):
    """depend of dt !!!!!"""               # need to change the computation !!
    N, W = Y
    indice_final_collapse = np.argmax(N < eps)
    if(indice_final_collapse==0):
        return np.NaN
    average = np.mean(W[:indice_final_collapse//2])
    indice_begin_collapse = len(N) - np.argmax((N > average)[::-1])
    if(indice_begin_collapse == len(N)):
        return np.NaN
    else:
        return (N[indice_final_collapse] - N[indice_begin_collapse]) / (indice_final_collapse - indice_begin_collapse)



def viability(Y, eps = eps):
    """depend of dt !!!!!"""               # need to change the computation !!
    N, W = Y
    i0 = 0
    if(N[-1] < eps): # collapse
        i3 = np.argmax(N < eps)
        average = np.mean(W[:i3//2])
        i1 = np.argmax(W < average)
        i2 = len(N) - np.argmax((N > average)[::-1])
        return (i2-i1)/(i3-i0)
    else:
        i3 = len(N)//2
        return np.NaN



def ratio(Y, eps = eps):
    """we assume we begin at the equilibrium, that the case in the study for now ..."""
    N, W = Y
    if(N[-1] < eps): # collapse
        i3 = np.argmax(N < eps)
    else:
        i3 = len(N)
    average = np.mean(W[:i3//2])
    return average/W[0]

    
    
    

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
    Variability_half = np.zeros_like(Collapse)
    Speed_collapse = np.zeros_like(Collapse)
    Viability = np.zeros_like(Collapse)
    Ratio = np.zeros_like(Collapse)

    for i in range(Number_of_simulation):
        O = Ode(**kwargs)
        Y = O.solve_by_part()
        # measures
        Collapse[i] = collapse(Y)
        Variability_always[i] = variability_always(Y)
        Variability_until[i] = variability_until(Y)
        Variability_only[i] = variability_only(Y)
        Variability_10[i] = variability_10(Y)
        Collapse_10_b[i] = collapse(Y[:,:len(Y[0])//10])
        Collapse_10_m[i] = Y[0,len(Y[0])//10]
        Variability_half = variability_half(Y)
        Speed_collapse = speed_collapse(Y)
        Viability = viability(Y)
        Ratio = ratio(Y)
    if(mean):
        return np.nanmean(Collapse), np.nanmean(Variability_always), np.nanmean(Variability_until), np.nanmean(Variability_only), np.nanmean(Variability_10), np.nanmean(Collapse_10_b), np.nanmean(Collapse_10_m), np.nanmean(Variability_half), np.nanmean(Speed_collapse), np.nanmean(Viability), np.nanmean(Ratio)
    else:
        return Collapse, Variability_always, Variability_until, Variability_only, Variability_10, Collapse_10_b, Collapse_10_m, Variability_half, Speed_collapse, Viability, Ratio