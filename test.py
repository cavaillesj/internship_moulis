#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:23:13 2019

@author: azer
"""

import numpy as np


Param_freq = {"p":0.01}
Param_ampl = {"scale":0.7}

Fire = {"frequence": "bernoulli",
        "param_freq" : Param_freq,
        "amplitude": "exponential",
        "param_amplitude" : Param_ampl}

Fire.items()

#print("Fire\nfrequence "+self.law_freq+" ("+str(0)+")\namplitude "+self.law_amplitude+"("+str(0)+")")

print("Fire"
      +"\nfrequence "+Fire["frequence"]+" "+str(Fire["param_freq"])
      +"\namplitude "+Fire["amplitude"]+" "+str(Param_ampl)) 


#### utilise pour le feux
      
plt.plot([1,2,3,4,5], [2,3,None,5,6])
plt.show()
