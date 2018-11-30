
'''Get feature data from file as a matrix with a row per data instance'''
import sys
def getFeatureData(featureFile):
    x=[]
    dFile = open(featureFile, 'r')
    i = 0
    for line in dFile:
        row = line.split()
        rVec = [float(item) for item in row]
        x.append(rVec)
    dFile.close()
    return x

'''Get label data from file as a dictionary with key as data instance index
and value as the class index
'''
def getLabelData(labelFile):
    lFile = open(labelFile, 'r')
    lDict = {}
    i = 0
    for line in lFile:
        row = line.split()
        lDict[int(row[1])] = int(row[0])
    lFile.close()
    return lDict

x = getFeatureData(sys.argv[1])
y_dict = getLabelData(sys.argv[2])
test_data = getFeatureData(sys.argv[3])

import random
train_size = 1.00
rand_index = random.sample(range(len(x)),int(train_size*len(x)))
train_x = []
train_y = []
test_x  = []
test_y  = []
for i in range(len(x)):
    if i in rand_index:
        train_x.append(x[i])
        train_y.append(y_dict[i])
    else:
        test_x.append(x[i])
        test_y.append(y_dict[i])

test_x = test_data

class_0 = []
class_1 = []
for i in range(len(train_x)):
    if train_y[i] == 0:
        class_0.append(train_x[i])
    else:
        class_1.append(train_x[i])

mean_c0 = [sum(dim)/len(dim) for dim in zip(*class_0)]
mean_c1 = [sum(dim)/len(dim) for dim in zip(*class_1)]

var_c0 = [sum([(class_0[i][j]-mean_c0[j])**2 for i in range(len(class_0))])/len(class_0) for j in range(len(class_0[0]))]
std_c0 = [vi**(1/2) for vi in var_c0]
var_c1 = [sum([(class_1[i][j]-mean_c1[j])**2 for i in range(len(class_1))])/len(class_1) for j in range(len(class_1[0]))]
std_c1 = [vi**(1/2) for vi in var_c1]

snr = [[abs((mean_c0[j]-mean_c1[j])/(std_c0[j]+std_c1[j])),j] for j in range(len(train_x[0]))]
snr = sorted(snr, key = lambda row: row[0], reverse = True)
top_snr = [pi[1] for pi in snr[:100]]

import math
def mi_score(u,v):
    mi = 0
    for ui in set(u):
        for vi in set(v):
            ui_vi = list(zip(u,v)).count((ui,vi))
            if ui_vi == 0:
                pass
            else:
                mi += (ui_vi/len(u))*math.log((len(u)*ui_vi)/(u.count(ui)*v.count(vi)))
    return mi

mi_array = [[j, mi_score([train_x[i][j] for i in range(len(train_x))], train_y)] for j in range(len(train_x[0]))]


mi_array = sorted(mi_array, key=lambda row: row[1], reverse=True)
top_mi = [pi[0] for pi in mi_array[:100]]

def mean(u):
    return sum(u)/len(u)

def fn_a(u):
    return (len(u)*sum([ui**2 for ui in u])-sum(u)**2)**(1/2)

def pearson_coef(u,v, mu_v, fn_a_v):
    mu_u = mean(u)
    fn_a_u = fn_a(u)
    return (sum([u[i]*v[i] for i in range(len(u))])-len(u)*mu_u*mu_v)/(fn_a(u)*fn_a(v))

pearson_array = []
mu_y_train = mean(train_y)
fn_a_y_train = fn_a(train_y)
for j in range(len(train_x[0])):
    pearson_array.append([j, pearson_coef([train_x[i][j] for i in range(len(train_x))], train_y, mu_y_train, fn_a_y_train)])

pearson = sorted(pearson_array, key = lambda row: row[1])
top_pearson = [pi[0] for pi in pearson[:100]]

common = []
for pi in top_mi:
    if pi in top_snr and top_pearson:
        common.append(pi)

red_x = []
red_x_test = []
for i in range(len(train_x)):
    red_x.append([train_x[i][j] for j in common[:24]])
for i in range(len(test_x)):    
    red_x_test.append([test_x[i][j] for j in common[:24]])

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

y_hats = []

clf = svm.SVC(kernel = "linear", C=2.0)
y_hat = clf.fit(red_x,train_y).predict(red_x_test)
#acc = sum([y_hat[i] == test_y[i] for i in range(len(red_x_test))])/len(red_x_test)
y_hats.append(y_hat)

clf = svm.SVC(kernel = "poly", degree = 2, C=2.0)
y_hat = clf.fit(red_x,train_y).predict(red_x_test)
#acc = sum([y_hat[i] == test_y[i] for i in range(len(red_x_test))])/len(red_x_test)
y_hats.append(y_hat)

clf = svm.SVC(kernel = "rbf", C=2.0)
y_hat = clf.fit(red_x,train_y).predict(red_x_test)
#acc = sum([y_hat[i] == test_y[i] for i in range(len(red_x_test))])/len(red_x_test)
y_hats.append(y_hat)

y_hat = []
for i in range(len(red_x_test)):
    if y_hats[0][i]+y_hats[1][i]+y_hats[2][i] < 1.5:
        y_hat.append(0)
    else:
        y_hat.append(1)

# used for optimizing
# acc = sum([y_hat[i] == test_y[i] for i in range(len(red_x_test))])/len(red_x_test)

for i in range(len(y_hat)):
    print(y_hat[i]," ",i)

print()
print("Feature #s used: ", common[:24])
