import sys

Number_elements = raw_input()

leng = int(Number_elements)
y = [0] * leng
k = [0] * leng
maths_list = [0] * leng
physics_list = [0] * leng
chemistry_list = [0] * leng

for i in range(leng):
    y[i] = raw_input()

for i in range(leng):
     k[i] = y[i].split()

for i in range(leng):
    maths_list[i] = k[i][0]
    physics_list[i] = k[i][1]
    chemistry_list[i] = k[i][2]

#
# print maths_list
#
# print physics_list
#
# print chemistry_list


from scipy.stats.stats import pearsonr


maths_listnumbers = map(int, maths_list)
physics_listnumbers = map(int, physics_list)
chemistry_listnumbers = map(int, chemistry_list)

MP = pearsonr(maths_listnumbers,physics_listnumbers)
PC = pearsonr(chemistry_listnumbers,physics_listnumbers)
MC = pearsonr(maths_listnumbers,chemistry_listnumbers)

print round(MP[0],2)
print round(PC[0],2)
print round(MC[0],2)



def Cor (list1,list2, len):

    sum_list1 = 0
    sum_list2 = 0
    sum_products = 0
    sum_list1square = 0
    sum_list2square = 0

    for i in range(len):
        sum_list1 = sum_list1 + list1[i]

    for i in range(len):
        sum_list2 = sum_list2 + list2[i]

    for i in range(len):
        sum_products = sum_products + list1[i]*list2[i]

    for i in range(len):
        sum_list1square = sum_list1square + list1[i]*list1[i]

    for i in range(len):
        sum_list2square = sum_list2square + list2[i]*list2[i]


    numerator =  (len * sum_products - sum_list1 * sum_list2)
    denominator1 = (len*sum_list1square - sum_list1 * sum_list1) ** 0.5
    denominator2 = (len*sum_list2square - sum_list2 * sum_list2) ** 0.5
    #
    # print sum_list1
    # print sum_list2
    # print sum_products
    # print sum_list1square
    # print sum_list2square

    coefficient = numerator / (denominator1 * denominator2)

    return coefficient


MP = round(Cor(maths_listnumbers,physics_listnumbers, leng),2)
PC = round(Cor(chemistry_listnumbers,physics_listnumbers, leng),2)
MC = round(Cor(maths_listnumbers,chemistry_listnumbers, leng),2)

print MP
print PC
print MC

#
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
#
#
