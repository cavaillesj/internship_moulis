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
