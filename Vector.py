import sys
import numpy as np

# a = np.array([[ 4, 2 ,0.6], [ 4.2, 2.1 ,0.59], [ 3.9, 2 ,0.58], [ 4.3, 2.1 ,0.62], [ 4.1, 2.2 ,0.63]])
# b =  np.transpose(a)
#
# c =  b.dot(a)
# print c
#
# d = np.cov(b)
#
# print d
#
# v = np.array([1,2,3])
# vtranspose = np.transpose(v)
# new1 =  vtranspose.dot(d)
#
# print new1
#
# new2 = new1.dot(v)
# print new2


#
# a = np.array([
#               [ 1, 2 ,4, 0, 0],
#               [ 4, 3 ,2, 0, 0],
#               [ 4, 2 ,2, 0, 0],
#               [ 3, 5 ,4, 0, 0],
#               [ 0, 0 ,0, 3, 4],
#               [ 0, 0 ,0, 3,5],
#               [ 0.0, 0 ,0, 5, 2]
#               ])


import os
import glob

os.chdir( '/Users/admin/Downloads' )

from numpy import genfromtxt
a = genfromtxt('SampleData - Sheet1.csv', delimiter=',')

# print a

# a.astype(float)
#
# print a
#
# print type(a)

from scipy.sparse.linalg import svds

u, s, v = svds(a, k=2)

# uT = np.transpose(u)
#
# print uT.dot(u)

# u, s, v = np.linalg.svd(a, full_matrices=False)

# print s


# u [ abs(u) < 0.1] = 0
u [ u < 0.1] = 0

print u

print s

v [ v < 0.1] = 0


print v


