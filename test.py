#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:23:13 2019

@author: azer
"""

import numpy as np

A = np.array([1,2,3,-1, 2,-2])
if((A < 0).any()):
    print("Extinction")
    


