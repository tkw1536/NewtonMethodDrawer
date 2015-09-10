import argparse
from cmath import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import colorsys


# Parses function arguments
def get_options():
    parser = argparse.ArgumentParser(description='Do some magic drawing of the Newton Method. ')

    g1 = parser.add_argument_group("Range")
    parser.add_argument("-d", "--degree", help="degree of equation to draw", type=int, default=3)

    crange = parser.add_argument_group("Coordinates")
    crange.add_argument("-xs", "--xmin", help="start of the X range", type=float, default=-1)
    crange.add_argument("-xe", "--xmax", help="end of the X range", type=float, default=1)
    crange.add_argument("-ys", "--ymin", help="start of the Y range", type=float, default=-1)
    crange.add_argument("-ye", "--ymax", help="start of the Y range", type=float, default=1)
    crange.add_argument("-x", "--xsteps", help="number of steps on the X axis", type=int, default=100)
    crange.add_argument("-y", "--ysteps", help="number of steps on the Y axis", type=int, default=100)

    return parser.parse_args()

# generates n distributed colors.
def get_colors(N):
    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    return list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

def make_f(roots):
    def f(z):
        return np.prod([z-r for r in roots])
    return f

def make_fp(roots):
    degrees = list(range(len(roots)))
    def fp(z):
        return np.sum([np.prod([z-r for r in roots[:i]+roots[i+1:]]) for i in degrees])
    return fp

def newton(f, df, z0, roots, max_iter = 1000, limit = 10e-3):

    # compute f(z_i), f'(f(z_i))
    zi = z0
    iterations = 0

    # conergent roots

    # we have a maximal number of iterations
    while iterations < max_iter:
        # and iterate by computing the new dz
        dz = (1.0*f(zi)) / (1.0*df(zi))
        zi = zi - dz

        # check if we are not changing anymore.
        if abs(dz) < limit:
            # check if we are sufficiently close to a root.
            for (index, root) in enumerate(roots):
                if abs(root - zi) < limit:
                    return (index, iterations)
        # we go to the next iteration
        iterations = iterations+1

    # we did not converge
    return (-1, iterations)

def crange(xmin, xmax, xstep, ymin, ymax, ystep):
    xs = np.linspace(xmin, xmax, xstep)
    ys = np.linspace(ymin, ymax, ystep)

    return (xs, ys, [(yi, xi, complex(x, y)) for (xi, x) in enumerate(xs) for (yi, y) in enumerate(ys)])

# Main Method.
def main():
    # Find the options
    options = get_options()

    # The coordinates we want to find.
    coords = []

    # Make a figure.
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot(111, aspect='equal')
    ax.axis([options.xmin, options.xmax, options.ymin, options.ymax])

    # generate a set of points.
    (xs, ys, points) = crange(options.xmin, options.xmax, options.xsteps, options.ymin, options.ymax, options.ysteps)
    colors = get_colors(options.degree)

    def start():

        plt.title("Drawing image, this might take a while. ")
        plt.draw()

        # compute the roots and funcitons.
        roots = [complex(*c) for c in coords]
        limit = 0.5*(options.xmax - options.xmin + options.ymax - options.ymin)
        f = make_f(roots)
        fp = make_fp(roots)

        # now iterate
        colormap = [[[0, 0, 0] for x in xs] for y in ys]

        for p in points:
            (x, y, z) = p
            (converges_to, iterations) = newton(f, fp, z, roots, max_iter = 4 * options.degree, limit = 10e-3)
            if converges_to >= 0:
                colormap[y][x] = list(colors[converges_to])

        ax.imshow(colormap, interpolation='bilinear', cmap=cm.RdYlGn, origin='upper', extent=[options.xmin, options.xmax, options.ymin, options.ymax])
        ax.axis([options.xmin, options.xmax, options.ymin, options.ymax])

        plt.title("")
        plt.draw()

    def plot_point(x, y):
        coords.append([x, y])
        ax.plot([x], [y], 'kx')
        plt.title("Drawing points: "+str(len(coords))+"/"+str(options.degree))
        plt.draw()

    def on_click(event):
        x = event.xdata
        y = event.ydata

        if x and y:
            plot_point(x, y)
        if len(coords) == options.degree:
            fig.canvas.mpl_disconnect(cid)
            start()

    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()




if __name__ == '__main__':
    main()
