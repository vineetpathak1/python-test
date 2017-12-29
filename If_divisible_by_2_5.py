


def sum_divided_by_3(n):
    divided_by_3_quotient = (n-1)/ 3
    sum = 3 * divided_by_3_quotient * ( divided_by_3_quotient + 1) / 2
    return sum

def sum_divided_by_5(n):
    divided_by_5_quotient = (n-1)/ 5
    sum = 5 * divided_by_5_quotient * ( divided_by_5_quotient + 1) / 2
    return sum

def sum_divided_by_15(n):
    divided_by_15_quotient = (n-1)/ 15
    sum = 15 * divided_by_15_quotient * ( divided_by_15_quotient + 1) / 2
    return sum

Number_test_cases = raw_input()

leng = int(Number_test_cases)


for i in range(leng):
    number_input = int(raw_input())

    sum = int( sum_divided_by_3(number_input)) + int(sum_divided_by_5(number_input)) - int (sum_divided_by_15(number_input))

    print sum
