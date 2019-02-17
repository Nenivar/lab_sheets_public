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
    
    def leastSquares(self):
        x, y = copy(self.xs), copy(self.ys)

        x.shape = (y.shape[0],1)
        col = np.array([1] * y.shape[0])
        col.shape = (y.shape[0],1)
        x = np.hstack((col, x))
        
        y = y.transpose()
        y.shape = (y.shape[0], 1)

        H = np.linalg.inv(np.matmul(x.T,x))
        J = np.matmul(np.matmul(H, x.T), y)
        return J
    
    def residual(self, a, b):
        return reduce(lambda acc, xy: acc + (xy[1] - (a + b * xy[0])) ** 2, zip(self.xs, self.ys), 0)

    def plot(self, ax):
        ax.plot(self.xs, self.ys)

# split into segments
segments = []
for i in range(0, len(xs), 20):
    segments.append(Segment(xs[i:i+20], ys[i:i+20]))

ax = plt.axes()

# for each segment...
for seg in segments:
    # determine the function type
    ans = seg.leastSquares()
    # produce the total reconstruction error
    err = seg.residual(ans[0][0], ans[1][0])
    print(err)
    # plot line if app.
    if plot:
        seg.plot(ax)

# produce a figure w/ reconstructed line
if plot:
    view_data_segments(xs, ys)