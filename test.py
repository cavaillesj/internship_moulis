#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:23:13 2019

@author: azer
"""


import numpy as np
import matplotlib.pyplot as plt

"""
Y, X = np.mgrid[-3:3:100j, -3:3:100j]
U = -1 - X**2 + Y
V = 1 + X - Y**2
speed = np.sqrt(U*U + V*V)



#fig0, ax0 = plt.subplots()
#strm = ax0.streamplot(X, Y, U, V, color=U, linewidth=2, cmap=plt.cm.autumn)
#fig0.colorbar(strm.lines)



fig1, ax1 = plt.subplots(ncols=1)
ax1.streamplot(X, Y, U, V, density=[0.5, 1])

#lw = 5*speed / speed.max()
#ax2.streamplot(X, Y, U, V, density=0.6, color='k', linewidth=lw)

plt.show()
"""


#Color = [["a"]*2]*2
#B = [[True, False], [False, True]]


X = np.linspace(0, 1, 10)
Y = np.linspace(0, 1, 10)
XX, YY = np.meshgrid(X, Y)
Z = XX+1
plt.contourf(XX, YY, Z)

