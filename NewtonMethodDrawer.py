import argparse
from cmath import *

import numpy as np
import matplotlib.pyplot as plt
import colorsys


# Parses function arguments
def get_options():
    parser = argparse.ArgumentParser(description='Do some magic drawing of the Newton Method. ')
    parser.add_argument("-d", "--degree", help="degree of equation to draw", type=int, default=9)
    return parser.parse_args()

# generates n distributed colors. 
def get_colors(N):
    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    return list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

def make_f(roots):
    complexroots = list(map(lambda z:complex(*z), roots))
    def f(z):
        return np.prod([z-r for r in complexroots])
    return f

def make_fp(roots):
    complexroots = list(map(lambda z:complex(*z), roots))
    degrees = range(len(complexroots))
    def fp(z):
        return np.sum([np.prod([z-r for r in complexroots[:i]+complexroots[i+1:]]) for i in degrees])
    return fp

# Main Method.     
def main():
    # Find the options
    options = get_options()
    
    # The coordinates we want to find. 
    coords = []
    colors = get_colors(options.degree)
    print(colors)
    
    # Make a figure. 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis([-1.0, 1.0, -1.0, 1.0])
    
    def start():
        f = make_f(coords)
        fp = make_fp(coords)
        print("f is ", fp)
        print(fp(complex(0, 0)))
        print("<done>")
        fig.canvas.mpl_disconnect(cid)
    
    def plot_point(x, y):
        ax.plot([x], [y], 'kx')
        plt.draw()
    
    def on_click(event):
        # Grab x and y coordinates
        x = event.xdata
        y = event.ydata
        
        if x and y:
            coords.append([x, y])
            plot_point(x, y)
        if len(coords) == options.degree:
            start()
    
    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    
    
    

if __name__ == '__main__':
    main()