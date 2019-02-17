from utilities import load_points_from_file, view_data_segments

import sys, os
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

# split into segments
segX = [xs[x:x+20] for x in range(0, len(xs), 20)]
segY = [ys[x:x+20] for x in range(0, len(xs), 20)]

Segment(segX[0], segY[0])

ax = plt.axes()

# for each segment...
for i in range(0, len(segX)):
    # determine the function type
    def leastSquares(x, y):
        x.shape = (y.shape[0],1)
        col = np.array([1] * y.shape[0])
        col.shape = (y.shape[0],1)
        x = np.hstack((col, x))
        
        y = y.transpose()
        y.shape = (y.shape[0], 1)

        H = np.linalg.inv(np.matmul(x.T,x))
        J = np.matmul(np.matmul(H, x.T), y)
        return J

    ans = leastSquares(segX[i], segY[i])

    # produce the total reconstruction error
    def residual(a, b, xs, ys):
        return reduce(lambda acc, xy: acc + (xy[1] - (a + b * xy[0][0])) ** 2, zip(xs, ys), 0)

    # produce a figure w/ reconstructed line
    if plot:
        ax.plot(segX[i], segY[i])

if plot:
    view_data_segments(xs, ys)