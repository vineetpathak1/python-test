# def rosen(x):
#      """The Rosenbrock function"""
#      return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)


def simplex2(x):
#      """The Rosenbrock function"""
      return (x-6)**2 + (x-1)**2
#      """return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)"""


# x0 = np.array([0])
#
# res = minimize(simplex2, x0, method='nelder-mead',options={'xtol': 1e-8, 'disp': True})
#
# print(res)

# A = A = np.array([[1,3,5],[2,5,1],[2,3,8]])
# B = linalg.inv(A)
#
# C = A.dot(B)
#
# D = linalg.det(C)
#
# print(D)
# print (C)
#
# la, v = linalg.eig(A)
# l1,l2,l3 = la
#
# print l1, l2, l3
#
# print v



# import matplotlib.pyplot as plt
#
# plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
# plt.plot(points[:,0], points[:,1], 'o')
#
# for j, p in enumerate(points):
#      plt.text(p[0]-0.03, p[1]+0.03, j, ha='right') # label the points
# for j, s in enumerate(tri.simplices):
#      p = points[s].mean(axis=0)
#      plt.text(p[0], p[1], '#%d' % j, ha='center') # label triangles
#
# plt.xlim(-0.5, 1.5); plt.ylim(-0.5, 1.5)
#
# plt.show()

#
# import sys
#
# x = raw_input()
# y = raw_input()
#
# k = y.split()
#
# leng = len(k)
#
# if int(x) != leng :
#      print ("Data size mismatch")
#
# numbers = map(int, k)
#
# import numpy as np
#
# from scipy.stats import mode
#
# rounded_mean = round(np.mean(numbers), 1)
# rounded_median = round(np.median(numbers), 1)
# rounded_mode = int(mode(numbers)[0][0])
# rounded_stddev= round(np.std(numbers), 1)
#
# rounded_lower_confidence_interval =  round(np.mean(numbers) -  1.96 * np.std(numbers)/leng ** 0.5, 1)
#
# rounded_upper_confidence_interval =  round(np.mean(numbers) +  1.96 * np.std(numbers)/leng ** 0.5, 1)
#
#
# print rounded_mean
# print rounded_median
# print rounded_mode
# print rounded_stddev
# print rounded_lower_confidence_interval, rounded_upper_confidence_interval





# import sys
#
# x = raw_input()
# y = raw_input()
#
# k = y.split()
#
# leng = len(k)
#
# if int(x) != leng :
#      print ("Data size mismatch")
#
# numbers = map(int, k)
#
# import numpy as np
#
# from scipy.stats import mode
#
# rounded_mean = round(np.mean(numbers), 1)
# rounded_median = round(np.median(numbers), 1)
# rounded_mode = int(mode(numbers)[0][0])
# rounded_stddev= round(np.std(numbers), 1)
#
# rounded_lower_confidence_interval =  round(np.mean(numbers) -  1.96 * np.std(numbers)/leng ** 0.5, 1)
#
# rounded_upper_confidence_interval =  round(np.mean(numbers) +  1.96 * np.std(numbers)/leng ** 0.5, 1)
#
#
# print rounded_mean
# print rounded_median
# print rounded_mode
# print rounded_stddev
# print rounded_lower_confidence_interval, rounded_upper_confidence_interval



x = int (2 ** 9.91)

print x


import math
y = round(math.log(170,2), 2)

print y
