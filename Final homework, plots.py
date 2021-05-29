# -*- coding: utf-8 -*-
"""
Created on Sat May 29 12:40:06 2021

@author: fredr
"""
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

def LL_energy(x,y=0,n=0):
    return (np.sign(n+0.1)*np.sqrt(x**2+y**2) + np.sign(n)*np.sqrt(2*np.abs(n)))

x = np.linspace(-6, 6, 100)
y = np.linspace(-6, 6, 100)

# setting the axes at the centre
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_ylim([-6,6])
for n in range(-10, 11):
    if n>=0:
        c = 'y'
    else:
        c = 'c'
    plt.plot(x,LL_energy(x,0,n), c)
plt.plot(x,(-1)*np.abs(x), 'c')
#plt.legend(["Landau levels for n between -10 and 10."], loc='upper center')
plt.show()
################# 3D plot
X, Y = np.meshgrid(x, y)
Z = LL_energy(X, Y, 0)

fig2 = plt.figure()
ax2 = plt.axes(projection='3d')
ax2.contour3D(X, Y, Z, 50, cmap='viridis',edgecolor='none')
#ax2.plot_wireframe(X, Y, Z, color='black')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z');
ax2.view_init(20, 35)
fig2