import sys


def all_same(items):
    return len(set(items)) == 1


numberTestCases = int(raw_input())


for i in range(numberTestCases):
    numberStudents = int(raw_input())

    GPADetails = raw_input()
    GPA_Student = [0] * numberStudents
    GPA_Student = GPADetails.split()

    Test1_Score = raw_input()
    Test1_Score_Student = [0] * numberStudents
    Test1_Score_Student = Test1_Score.split()

    Test2_Score = raw_input()
    Test2_Score_Student = [0] * numberStudents
    Test2_Score_Student = Test2_Score.split()


    Test3_Score = raw_input()
    Test3_Score_Student = [0] * numberStudents
    Test3_Score_Student = Test3_Score.split()

    Test4_Score = raw_input()
    Test4_Score_Student = [0] * numberStudents
    Test4_Score_Student = Test4_Score.split()

    Test5_Score = raw_input()
    Test5_Score_Student = [0] * numberStudents
    Test5_Score_Student = Test5_Score.split()

    GPA_Student_Float = map(float, GPA_Student)

    Test1_Score_Student_Float = map(float, Test1_Score_Student)
    Test2_Score_Student_Float = map(float, Test2_Score_Student)
    Test3_Score_Student_Float = map(float, Test3_Score_Student)
    Test4_Score_Student_Float = map(float, Test4_Score_Student)
    Test5_Score_Student_Float = map(float, Test5_Score_Student)

    Test_Correlation = [0] * 5

    for i in range(5):
        Test_Correlation[1]

    from scipy.stats.stats import pearsonr

    Test1_Boolean = all_same(Test1_Score_Student_Float)
    if  Test1_Boolean == False :
        Test_Correlation[0] = pearsonr(GPA_Student_Float,Test1_Score_Student_Float)[0]
    else :
        Test_Correlation[0] = -1

    Test2_Boolean = all_same(Test2_Score_Student_Float)
    if  Test2_Boolean == False :
        Test_Correlation[1] = pearsonr(GPA_Student_Float,Test2_Score_Student_Float)[0]
    else :
        Test_Correlation[1] = -1

    Test3_Boolean = all_same(Test3_Score_Student_Float)
    if  Test3_Boolean == False :
        Test_Correlation[2] = pearsonr(GPA_Student_Float,Test3_Score_Student_Float)[0]
    else :
        Test_Correlation[2] = -1

    Test4_Boolean = all_same(Test4_Score_Student_Float)
    if  Test4_Boolean == False :
        Test_Correlation[3] = pearsonr(GPA_Student_Float,Test4_Score_Student_Float)[0]
    else :
        Test_Correlation[3] = -1

    Test5_Boolean = all_same(Test5_Score_Student_Float)
    if  Test5_Boolean == False :
        Test_Correlation[4] = pearsonr(GPA_Student_Float,Test5_Score_Student_Float)[0]
    else :
        Test_Correlation[4] = -1

    # print Test_Correlation

    import operator

    largest_value = max(Test_Correlation)
    max_index = Test_Correlation.index(largest_value)

    print max_index + 1

