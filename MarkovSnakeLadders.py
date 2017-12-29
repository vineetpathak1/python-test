
Number_elements = raw_input()

leng = int(Number_elements)


import math
#
# def prime_factors(n):
#     i = 2
#     factors = []
#     while i * i <= n:
#         if n % i:
#             i += 1
#         else:
#             n //= i
#             factors.append(i)
#     if n > 1:
#         factors.append(n)
#     return factors
#
#
# def if_terminating(a,b):
#     c = []
#     for item in b:
#         if item not in a:
#             c.append(item)
#         else:
#             a.remove(item)
#
#     set_b = set(c)
#     set_b.discard(2)
#     set_b.discard(5)
#     len_set = set_b.__len__()
#     if len_set > 0:
#         ret_terminating = False
#     else:
#         ret_terminating = True
#
#     return ret_terminating


def if_any_factor_other_than_2_5(n):
    i = 2
    while (n % i == 0):
        n = n/i

    i = 5
    while (n % i == 0):
        n = n/i

    if n > 1:
        ret_terminating = False
    else:
        ret_terminating = True


    return ret_terminating



e = math.e

from fractions import gcd

def if_terminating_by_gcd(num, divisor):
    gcd1 = gcd(num, divisor)
    updated_divisor = divisor/ gcd1

    ret_terminating = if_any_factor_other_than_2_5(updated_divisor)
    return ret_terminating
    # denominator_factor = prime_factors(updated_divisor)
    # set_b = set(denominator_factor)
    # set_b.discard(2)
    # set_b.discard(5)
    # len_set = set_b.__len__()
    # if len_set > 0:
    #     ret_terminating = False
    # else:
    #     ret_terminating = True
    #
    # return ret_terminating


max_input_number = 5
input_list = []
resultList = []

for i in range(leng):
     number_input = int(raw_input())
     input_list.append(number_input)
     if number_input > max_input_number:
         max_input_number = number_input

sum = 0

number_input = 100000

resultList[0:4] = [0,0,0,0,0]

current_max_divisor = 2
current_max_divisor_factor = [2]

for i in range (5, max_input_number +1):
    f = int(i)/ e
    max_divisor = int(round(f))
    # numerator_factor = prime_factors(i)
    # if max_divisor > current_max_divisor:
    #     denominator_factor = prime_factors(max_divisor)
    #     current_max_divisor =   max_divisor
    #     current_max_divisor_factor =  denominator_factor
    # else :
    #     denominator_factor = current_max_divisor_factor
    # # IfTerminate = if_terminating(numerator_factor,denominator_factor)
    IfTerminate = if_terminating_by_gcd(i,max_divisor)

    if IfTerminate:
        incremental =  (-1) * i
    else :
        incremental = i

    sum = int(sum) + incremental

    # print i, max_divisor, numerator_factor, denominator_factor, IfTerminate, sum

    resultList.append(sum)

# print sum

# print resultList[10000]



for i in range(leng):
     print resultList[input_list[i]]



#  Following code works correctly. However, not performance efficinet if number of inouts are high as the calculations are duplicated. Hence, plan to optimize
# for i in range(leng):
#     number_input = int(raw_input())
#
#     sum = 0
#
#     # number_input = 1000000
#     for i in range (5, number_input +1):
#         f = int(i)/ e
#         max_divisor = int(round(f))
#         numerator_factor = prime_factors(i)
#         denominator_factor = prime_factors(max_divisor)
#         IfTerminate = if_terminating(numerator_factor,denominator_factor)
#
#         if IfTerminate:
#             incremental =  (-1) * i
#         else :
#             incremental = i
#
#         sum = int(sum) + incremental
#
#         # print i, max_divisor, numerator_factor, denominator_factor, IfTerminate, sum
#
#     print sum


# number_input = int(raw_input())

