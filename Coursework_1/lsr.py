from utilities import load_points_from_file, view_data_segments

import sys, os
from copy import copy, deepcopy
from functools import reduce, partial

import numpy as np
import math
from matplotlib import pyplot as plt

# read in options
file = sys.argv[1]
plot = False
if len(sys.argv) >= 3:
    plot = True if sys.argv[2] == '--plot' else False
xs, ys = load_points_from_file(file)

# returns whatever you give it :)
identity = lambda x: x

# returns y val for a point x
# for a poly a + bf(X) + cf(X)^2 + ...
def getLineVal(coeff, f, x):
    y = 0
    for deg in range(0, len(coeff)):
        y += coeff[deg] * (f(x) ** deg)
    return y

class Segment():
    def __init__(self, xs, ys):
        self.xs, self.ys = xs, ys
    
    # returns coeff calculated from least squares
    # for a poly of degree p (a + bf(X) + cf(X)^2 + ...)
    def leastSquaresPoly(self, p, func=identity):
        X = np.vander(self.xs, p + 1, True)
        if X.shape[1] > 0:
            for col in range(1, X.shape[1]):
                X[:,col] = [ func(x) for x in X[:,col] ]
        
        # (X^t * X)^-1 * X^T * y
        H = np.linalg.inv(np.matmul(X.T, X))
        C = np.matmul(X.T, self.ys)
        coeff = np.matmul(H, C)
        return coeff
    
    # returns error between line & actual points
    # func :: x -> y
    def error(self, f):
        err = 0
        for i in range(0, len(self.xs)):
            x = self.xs[i]
            y = self.ys[i]
            err += (y - f(x)) ** 2
        return err
    
    # plots a segment on a given axis
    # for a given function
    def plot(self, ax, func):
        width = np.linspace(self.xs[0], self.xs[len(self.xs) - 1], 50)
        yVals = list(map(func, list(width)))
        ax.plot(width, yVals)

# split into segments
segments = []
for i in range(0, len(xs), 20):
    segments.append(Segment(xs[i:i+20], ys[i:i+20]))
ax = plt.axes()

# for each segment...
totErr = 0
for seg in segments:
    # line = (func, poly. degree)
    lines = []
    lines.append((np.sin, 1))
    lines.extend( [ (identity, p) for p in [1, 3] ] )

    # create functions & errors
    # for each line type
    funcs = {}
    err = {}
    for f in lines:
        coeff = seg.leastSquaresPoly(f[1], f[0])
        funcs[f] = partial(getLineVal, coeff, f[0])
        err[f] = seg.error(funcs[f])    

    # find the line type with minimal error
    minErr = min(err, key=err.get)
    # if we can go down to p=1 we should
    if minErr != (identity, 1):
        if math.isclose(err[minErr], err[(identity, 1)], rel_tol=0.15):
            minErr = (identity, 1)
    
    #print(err)
    #print('=>', minErr)

    totErr += err[minErr]

    # plot line if app.
    if plot:
        seg.plot(ax, funcs[minErr])

# print total error
print(totErr)

# produce a figure w/ reconstructed line
if plot:
    view_data_segments(xs, ys)