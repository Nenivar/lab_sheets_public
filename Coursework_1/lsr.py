from utilities import load_points_from_file, view_data_segments

import sys, os
from copy import copy, deepcopy
from functools import reduce

import numpy as np
from matplotlib import pyplot as plt

# read in options
file = sys.argv[1]
plot = False
if len(sys.argv) >= 3:
    plot = True if sys.argv[2] == '--plot' else False
xs, ys = load_points_from_file(file)

class Segment():
    def __init__(self, xs, ys):
        self.xs, self.ys = xs, ys
    
    def leastSquaresPoly(self, p):
        X = np.empty((0, 20))
        for deg in range(0, p + 1):
            row = list(map(lambda x: x ** deg, self.xs))
            X = np.insert(X, deg, row, 0)
        
        # (X^t * X)^-1 * X^T * y
        H = np.linalg.inv(np.matmul(X, X.T))
        C = np.matmul(X, self.ys)
        coeff = np.matmul(H, C)
        return coeff
    
    def error(self, coeff):
        err = 0
        for i in range(0, len(self.xs)):
            x = self.xs[i]
            y = self.ys[i]
            lineY = 0
            for deg in range(0, len(coeff)):
                lineY += coeff[deg] * (x ** deg)
            err += abs(y - lineY)
        return err
    
    def residual(self, a, b):
        return reduce(lambda acc, xy: acc + (xy[1] - (a + b * xy[0])) ** 2, zip(self.xs, self.ys), 0)
    
    def plotPoly(self, ax, coeff):
        width = np.linspace(self.xs[0], self.xs[len(self.xs) - 1], 50)
        yVals = 0
        for i in range(0, len(coeff)):
            yVals += coeff[i] * (width ** i)
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
    for p in range(1, 5):
        ans = seg.leastSquaresPoly(p)
        errs[p] = seg.error(ans)
    minErr = min(errs, key=errs.get)
    #print(errs)
    print('=>', minErr)
    ans = seg.leastSquaresPoly(minErr)

    # produce the total reconstruction error
    ##err = seg.residual(ans[0], ans[1])

    # plot line if app.
    if plot:
        seg.plotPoly(ax, ans)

# produce a figure w/ reconstructed line
if plot:
    view_data_segments(xs, ys)