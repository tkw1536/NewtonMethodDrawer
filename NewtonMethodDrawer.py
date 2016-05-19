import argparse
from cmath import *

import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import colorsys


def get_options():
    """
        Parses all the options given to the program.
    """
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

    it = parser.add_argument_group("Iteration")
    it.add_argument("-i", "--iterations", help="Maximal number of iterations. -1 means auto-select according to degree. ", type=int, default=-1)
    it.add_argument("-e", "--epsilon", help="Radius of ball considdered to be converging to a certain root. ", type=float, default=10e-3)

    # Parse the arguments
    options = parser.parse_args()

    # Set some defaults
    if options.iterations == -1:
        options.iterations = 4 * options.degree

    # TODO: Implement a GUI.

    return options


# generates a color distribution
def get_colors(N):
    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    return list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

# Make a function from the roots
def make_f(roots):
    #write some code for the function
    func_code = "*".join(["(z-"+str(r)+")" for r in roots])

    # execute this and return it
    return eval("lambda z:(%s)" % (func_code, ))

def make_fp(roots):

    # we need to cache the function code to be faster
    func_code = "+".join(
        [
            "(" +
                "*".join(["(z-"+str(r)+")" for r in roots[:i]+roots[i+1:]])
            + ")"
            for i in range(len(roots))
        ]
    )

    return eval("lambda z:(%s)" % (func_code, ))

def newton(f, df, z0, roots, max_iter, limit):

    # compute f(z_i), f'(f(z_i))
    zi = z0
    iterations = 0

    # we have a maximal number of iterations
    while iterations < max_iter:
        # and iterate by computing the new dz
        dz = (1.0*f(zi)) / (1.0*df(zi))
        zi = zi - dz

        # if our change is small enough, we need to stop.
        if abs(dz) < limit:

            # try and find the root that we are closest to.
            for (index, root) in enumerate(roots):
                if abs(root - zi) < limit:
                    return (index, iterations)

        # if that did not work, go to the next iteration.
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
    fig = plt.figure()
    fig.canvas.set_window_title('NewtonMethodDrawer')
    ax = fig.add_subplot(111, aspect='equal')
    ax.axis([options.xmin, options.xmax, options.ymin, options.ymax])

    # generate a set of points.
    (xs, ys, points) = crange(options.xmin, options.xmax, options.xsteps, options.ymin, options.ymax, options.ysteps)
    colors = get_colors(options.degree)

    def start():

        plt.title("Computing image, this might take a while. ")
        plt.draw()

        # compute the roots and funcitons.
        roots = [complex(*c) for c in coords]
        limit = 0.5*(options.xmax - options.xmin + options.ymax - options.ymin)
        f = make_f(roots)
        fp = make_fp(roots)

        # now iterate
        colormap = [[[0, 0, 0] for x in xs] for y in ys]

        # now run it for all the point
        for (x, y, z) in points:
            (converges_to, iterations) = newton(f, fp, z, roots, max_iter = options.iterations, limit = options.epsilon)
            if converges_to >= 0:
                colormap[y][x] = list(colors[converges_to])

        ax.imshow(colormap, interpolation='bilinear', cmap=cm.RdYlGn, origin='upper', extent=[options.xmin, options.xmax, options.ymin, options.ymax])
        ax.axis([options.xmin, options.xmax, options.ymin, options.ymax])

        # finally update the map
        plt.title("")
        plt.draw()

    def on_click(event):
        # grab the event coordinates
        x = event.xdata
        y = event.ydata

        if x and y:
            # store the roots, update the heading and plot.
            coords.append([x, y])
            ax.plot([x], [y], 'kx')
            plt.title("Selecting points: "+str(len(coords))+"/"+str(options.degree))
            plt.draw()

        # if we are done, we shoudl get started.
        if len(coords) == options.degree:
            fig.canvas.mpl_disconnect(cid)

            # TODO: Run this in a seperate thread and return later.
            start()

    # register an event handler
    cid = fig.canvas.mpl_connect('button_press_event', on_click)

    # set the title
    plt.title("Selecting points: "+str(len(coords))+"/"+str(options.degree))

    # and get started.
    plt.show()


if __name__ == '__main__':
    main()
else:
    print("NewtonMethodDrawer can not be imported. ")
    sys.exit(-1)
