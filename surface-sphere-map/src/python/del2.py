# FOR TESTING ONLY
# http://pastebin.com/5A3fj0p7
#image = [3 4 6 7; 8 9 10 11; 12 13 14 15;16 17 18 19]
#del2(image)
#       
#0.25000  -0.25000  -0.25000  -0.75000
#  -0.25000  -0.25000   0.00000   0.00000
#   0.00000   0.00000   0.00000   0.00000
#   0.25000   0.25000   0.00000   0.00000
#       
#import numpy as np
#from scipy import ndimage
#import scipy.ndimage.filters
# 
#image =  np.array([[3, 4, 6, 7],[8, 9, 10, 11],[12, 13, 14, 15],[16, 17, 18, 19]])
#stencil = np.array([[0, 1, 0],[1, -4, 1], [0, 1, 0]])
#print ndimage.convolve(image, stencil, mode='wrap')
#       
#[[ 23  19  15  11]
# [  3  -1   0  -4]
# [  4   0   0  -4]
# [-13 -17 -16 -20]]
#       
#scipy.ndimage.filters.laplace(image)
#       
#[[ 6  6  3  3]
# [ 0 -1  0 -1]
# [ 1  0  0 -1]
# [-3 -4 -4 -5]]
#       
#from scipy.ndimage import convolve
#stencil= (1.0/(12.0*dL*dL))*np.array(
#        [[0,0,-1,0,0],
#         [0,0,16,0,0],
#         [-1,16,-60,16,-1],
#         [0,0,16,0,0],
#         [0,0,-1,0,0]])
#convolve(e2, stencil, mode='wrap')

from __future__ import division
       
import numpy as np
 
def del2(M):
    dx = 1
    dy = 1
    rows, cols = M.shape
    dx = dx * np.ones ((1, cols - 1))
    dy = dy * np.ones ((rows-1, 1))
 
    mr, mc = M.shape
    D = np.zeros ((mr, mc))
 
    if (mr >= 3):
        ## x direction
        ## left and right boundary
        D[:, 0] = (M[:, 0] - 2 * M[:, 1] + M[:, 2]) / (dx[:,0] * dx[:,1])
        D[:, mc-1] = (M[:, mc - 3] - 2 * M[:, mc - 2] + M[:, mc-1]) \
            / (dx[:,mc - 3] * dx[:,mc - 2])
 
        ## interior points
        tmp1 = D[:, 1:mc - 1]
        tmp2 = (M[:, 2:mc] - 2 * M[:, 1:mc - 1] + M[:, 0:mc - 2])
        tmp3 = np.kron (dx[:,0:mc -2] * dx[:,1:mc - 1], np.ones ((mr, 1)))
        D[:, 1:mc - 1] = tmp1 + tmp2 / tmp3
 
    if (mr >= 3):
        ## y direction
        ## top and bottom boundary
        D[0, :] = D[0,:]  + \
            (M[0, :] - 2 * M[1, :] + M[2, :] ) / (dy[0,:] * dy[1,:])
 
        D[mr-1, :] = D[mr-1, :] \
                   + (M[mr-3,:] - 2 * M[mr-2, :] + M[mr-1, :]) \
                   / (dy[mr-3,:] * dx[:,mr-2])
 
        ## interior points
        tmp1 = D[1:mr-1, :]
        tmp2 = (M[2:mr, :] - 2 * M[1:mr - 1, :] + M[0:mr-2, :])
        tmp3 = np.kron (dy[0:mr-2,:] * dy[1:mr-1,:], np.ones ((1, mc)))
        D[1:mr-1, :] = tmp1 + tmp2 / tmp3
 
    return D / 4
