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
        # (X^t * X)^-1 * X^T * y

        X = np.empty((0, 20))
        for deg in range(0, p + 1):
            row = list(map(lambda x: x ** deg, self.xs))
            X = np.insert(X, deg, row, 0)
        
        #print(X)
        #print(self.ys)
        H = np.linalg.inv(np.matmul(X, X.T))
        #print(H)
        #print(H.shape)
        C = np.matmul(X, self.ys)
        #print(C)
        #print(H.shape, C.shape)
        R = np.matmul(H, C)
        #print(R)

        coeff = []
        return R
    
    def leastSquares(self):
        x, y = copy(self.xs), copy(self.ys)

        x.shape = (y.shape[0],1)
        col = np.array([1] * y.shape[0])
        col.shape = (y.shape[0],1)
        x = np.hstack((col, x))
        
        y = y.transpose()
        y.shape = (y.shape[0], 1)

        H = np.linalg.inv(np.matmul(x.T,x))
        ab = np.matmul(np.matmul(H, x.T), y)
        return ab
    
    def residual(self, a, b):
        return reduce(lambda acc, xy: acc + (xy[1] - (a + b * xy[0])) ** 2, zip(self.xs, self.ys), 0)

    def plot(self, ax, a, b):
        #ax.plot(self.xs, self.ys)
        width = np.linspace(self.xs[0], self.xs[len(self.xs) - 1], 50)
        ax.plot(width, b * width + a)
    
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
    ans = seg.leastSquaresPoly(3)
    #ansP = seg.leastSquaresPoly(1)
    #print('A & B', ans, ansP)

    # produce the total reconstruction error
    err = seg.residual(ans[0], ans[1])
    print(err)
    # plot line if app.
    if plot:
        #seg.plot(ax, ans[0], ans[1])
        seg.plotPoly(ax, ans)

# produce a figure w/ reconstructed line
if plot:
    view_data_segments(xs, ys)