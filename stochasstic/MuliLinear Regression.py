import sys

Features_Training_Data = raw_input()

feature_count = int(Features_Training_Data.split()[0])
training_data_count = int(Features_Training_Data.split()[1])

leng = int(training_data_count)

y = [0] * leng


for i in range(training_data_count):
    y[i] = raw_input()

test_data_count = int(raw_input())

test = [0] * test_data_count

for i in range(test_data_count):
    test[i] = raw_input()



k = [0] * leng

for i in range(leng):
     k[i] = y[i].split()

featurelist = []
# #
#
for i in range(feature_count):
    featurelist.append([])


for f in range(feature_count):
     for i in range(leng):
           featurelist[f].append(k[i][f])
# #


# from sklearn.linear_model import LinearRegression

#import statsmodels.formula.api as sm

import statsmodels.api as sm

import pandas as pd
import numpy as np

labels = []
for i in range(feature_count):
    labels.append('F' + str(i+1))

labels.append('Y')


# labels = ['F1', 'F2', 'F3']

df = pd.DataFrame.from_records(k, columns=labels)

# X = df[["F1", "F2" ]]
Y = df[["Y" ]]

Z = []
Q = []
for i in range(feature_count):
    Z.append("F" + str(i+1))


Q.append(Z)

print Q
X = df[Q[0]]

Y = df[["Y" ]]

X = sm.add_constant(X)



result = sm.OLS(Y.astype(float), X.astype(float)).fit()
# print result.summary()

#
# TestData = [{'F1': 0.49, 'F2': 0.18}]
# dfTest = pd.DataFrame(TestData)
# dfTest = sm.add_constant(dfTest)
# print dfTest



t = [0] * test_data_count

for i in range(test_data_count):
     t[i] = test[i].split()

labels = []
for i in range(feature_count):
    labels.append('F' + str(i+1))
# labels = ['F1', 'F2']
df = pd.DataFrame.from_records(t, columns=labels)
#
X = df[Q[0]]

# X = df[["F1", "F2" ]]
#
X = sm.add_constant(X)
# print X
#
predictions = result.predict(X.astype(float))

for i in range(test_data_count):
    print round(predictions [i] + 0.005, 2)

# TestData = [{'F1': 0.49, 'F2': 0.18}]
# dfTest = pd.DataFrame(TestData)
# predictions = result.predict(dfTest)
# print predictions
