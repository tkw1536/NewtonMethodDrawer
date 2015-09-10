import colorsys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import math as mth



print("Root plotter for Dierk Schleicher")
print("Really simple version")

# The function used to generate the fractal.
def f(x):
    return ((x)**(3)-1)

# The derivative of the above function.
def fd(x):
    return (3*x**(2))
    
def roots():
    return np.array([1,0,0,-1])
    
"""
step = 0.00390625
"""
step = 0.00390625
multval = int(1/step)

rootar = np.roots(roots())
root1 = rootar[0]
root2 = rootar[1]
root3 = rootar[2]

newx = []
newy = []
newz = []
newr = []
colors = [[0 for x in range(multval*2)] for x in range(multval*2)]

def newton(f, fd, x):
    f_temp = f(x)
    fd_temp = fd(x)
    xk = 0
    while 1:
        if ((fd_temp)==0):
            newr.append(0)
            return 0
        dx = f_temp / (fd_temp)
        if abs(dx) < 10e-3 * (1 + abs(x)):
            if(np.abs((root1-(x-dx)))<10e-3):
                rootval = 1
            if(np.abs((root2-(x-dx)))<10e-3):
                rootval = 2
            if(np.abs((root3-(x-dx)))<10e-3):
                rootval = 3
            newr.append(rootval)
            return rootval
        x = x - dx
        f_temp = f(x)
        fd_temp = fd(x)
        xk = xk + 1

def frange2(start, stop, step, r):
    l = start
    while l < stop:
        newx.append(r)
        newy.append(l)
        z = complex(r,l)
        newz.append(z)
        newton(f,fd,z)
        l += step

def frange(start, stop, step):
    r = start
    count = 0
    while r < stop:
        frange2(start, stop, step, r)
        count += 1
        r += step

frange(-1,1,step)

rangeval = multval*4*multval        

for i in range(0,rangeval):
        x_value = int((newx[i]+1)*multval)
        y_value = int((newy[i]+1)*multval)
        if(newr[i]==0):
            colors[y_value][x_value] = [0.0, 0.0, 0.0]
        if(newr[i]==1):
            colors[y_value][x_value] = [1.0, 0.0, 0.0]
        if(newr[i]==2):
            colors[y_value][x_value] = [0.0, 1.0, 0.0]
        if(newr[i]==3):
            colors[y_value][x_value] = [0.0, 0.0, 1.0]

delta = step
x_g = y_g = np.arange(-1.0, 1.0, delta)
X, Y = np.meshgrid(x_g, y_g)

fig = plt.figure(1)
plt = fig.add_subplot(111)
plt.imshow(colors, interpolation='bilinear', cmap=cm.RdYlGn, origin='upper', extent=[-1,1,-1,1])

def onclick(event):
    print(event.button, event.x, event.y, event.xdata, event.ydata)

fig.canvas.mpl_connect('button_press_event', onclick)
fig.show()