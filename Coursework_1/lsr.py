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

def itself(x):
    return x

# returns y val for a point x
# for a poly a + bX + cX^2 + ...
def getLineVal(coeff, x):
    y = 0
    for deg in range(0, len(coeff)):
        y += coeff[deg] * (x ** deg)
    return y

class Segment():
    def __init__(self, xs, ys):
        self.xs, self.ys = xs, ys
    
    def leastSquaresPoly(self, p, func=itself):
        X = np.empty((0, 20))
        for deg in range(0, p + 1):
            #row = list(map(lambda x: x ** deg, func(self.xs)))
            row = list(map(lambda x: x ** deg, func(self.xs)))
            X = np.insert(X, deg, row, 0)
        
        X = np.vander(self.xs, p + 1, True)
        if X.shape[1] > 0:
            for col in range(1, X.shape[1]):
                X[:,col] = [ func(x) for x in X[:,col] ]
        #print(X.shape[1])
        #X[:,1] = [ np.sin(x) for x in X[:,1] ]
        #X[:,2] = [ np.sin(x) for x in X[:,2] ]
        #X[:,1] = [ np.sin(x) for x in X[:,1] ]
        #print(X)
        
        # (X^t * X)^-1 * X^T * y
        """ H = np.linalg.inv(np.matmul(X, X.T))
        C = np.matmul(X, self.ys)
        coeff = np.matmul(H, C) """
        H = np.linalg.inv(np.matmul(X.T, X))
        C = np.matmul(X.T, self.ys)
        coeff = np.matmul(H, C)
        return coeff
    
    def sinn(self):
        np.vander(self.xs, 1, increasing=True)
    
    # returns error between line & actual points
    # func :: x -> y
    def errorF(self, func):
        err = 0
        for i in range(0, len(self.xs)):
            x = self.xs[i]
            y = self.ys[i]
            err += abs(y - func(x))
        return err
    
    def error(self, coeff):
        err = 0
        for i in range(0, len(self.xs)):
            x = self.xs[i]
            y = self.ys[i]
            lineY = getLineVal(coeff, x)
            #for deg in range(0, len(coeff)):
                #lineY += coeff[deg] * (x ** deg)
            err += abs(y - lineY)
        #return abs(err)
        #return abs(err)
        return err
    
    def residual(self, a, b):
        return reduce(lambda acc, xy: acc + (xy[1] - (a + b * xy[0])) ** 2, zip(self.xs, self.ys), 0)
    
    def regularize(self, coeff):
        print(coeff, reduce(lambda acc, c: acc + c ** 2, coeff))
        return reduce(lambda acc, c: acc + c ** 2, coeff)
        #return reduce(lambda acc, c: abs(acc + c), coeff)
    
    def plotPoly(self, ax, coeff):
        width = np.linspace(self.xs[0], self.xs[len(self.xs) - 1], 50)
        yVals = 0
        for i in range(0, len(coeff)):
            yVals += coeff[i] * (width ** i)
        #print(yVals)
        ax.plot(width, yVals)
    
    def plotF(self, ax, func):
        width = np.linspace(self.xs[0], self.xs[len(self.xs) - 1], 50)
        yVals = list(map(func, list(width)))
        #yVals = list(map(lambda x: x + 1, width))

        #for i in range(0, len(coeff)):
            #yVals += coeff[i] * (width ** i)
        #print(yVals)
        ax.plot(width, yVals)

# split into segments
segments = []
for i in range(0, len(xs), 20):
    segments.append(Segment(xs[i:i+20], ys[i:i+20]))
ax = plt.axes()

# for each segment...
for seg in segments:
    # determine the function type
    ans = []
    errs = {}
    for p in [1,3]:#range(1, 3):
        ans = seg.leastSquaresPoly(p)
        #seg.regularize(ans)
        #errs[p] = seg.error(ans) ** 2 - 5 * seg.regularize(ans)
        errs[p] = seg.error(ans)

    ans = seg.leastSquaresPoly(2, np.sin)
    #print(ans)
    errs[np.sin] = seg.error(ans)
    #print(errs)

    minErr = min(errs, key=errs.get)
    #print(errs)
    #print('=>', minErr)
    """ if minErr == np.sin:
        ans = seg.leastSquaresPoly(2, np.sin)
    else:
        ans = seg.leastSquaresPoly(minErr) """
    ans = seg.leastSquaresPoly(2, np.sin)

    # produce the total reconstruction error
    ##err = seg.residual(ans[0], ans[1])

    funcs = {}

    funcs['sin'] = lambda x: np.sin(x)
    for i in [1, 3]:
        coeff = seg.leastSquaresPoly(i)
        funcs[i] = partial(getLineVal, coeff)
    
    errs = {}
    for f in funcs.values():
        errs[f] = seg.errorF(f)
        print(errs[f])
    minErr = min(errs, key=errs.get)
    print(funcs)

    # plot line if app.
    if plot:
        #seg.plotPoly(ax, ans)
        seg.plotF(ax, minErr)

# produce a figure w/ reconstructed line
if plot:
    view_data_segments(xs, ys)