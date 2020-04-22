import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

from utils import *


###################################################
# Load Data
###################################################
data = loadmat('test_train_data_class3.mat')
data_train = np.array(data['Data']['train'][0][0][0])
data_test  = np.array(data['Data']['test'][0][0][0])
dim = data_train.shape[0]

# for c in data_train[0]:
#     print_c('',c)
# input()



###################################################
# a) Maximum likelihood estimator
###################################################
# Compute mean
class_means = []
for d in data_train:
    class_means.append(mean_mv(d))

# Compute covariance
class_covs = []
for d in data_train:
    class_covs.append(cov(d))

print_c("Means:",class_means)
print_c("Covariances:",class_covs)



###################################################
# c) Classify test points
###################################################
# Compute Scores using g_1b
test_scores = []
for k in range(len(data_test)):
    for i in range(len(data_test[k][0])):
        gi = []
        for j in range(len(data_train)):
            gi.append(g_1b_mv(np.array(data_test[k])[:,i],data_train[j],.33))
        test_scores.append(np.array(gi))

test_scores = np.array(test_scores)
#print_c('Scores for test samples:',test_scores)


# Classify
predictions = []
for ts in test_scores:
    predictions.append(classifier(ts))
predictions = np.array(predictions)

#print('Classification:',predictions)


# Build confusion matrix
## Build labels
labels = []
for i,d in enumerate(data_test):
    for x in d[0]:
        labels.append(i)
labels = np.array(labels)

## Build CM
cm = np.zeros((dim,dim))
good = 0
for pred,lab in zip(predictions,labels):
    cm[pred,lab] += 1

## Compute test score
cm_score = 0.0
for i in range(dim):
    cm_score += cm[i,i]
cm_score = round(cm_score / len(labels), 2)

print_c('Confusion Matrix:\nScore: '+str(cm_score),cm)



###################################################
# d) Bayesian Estimates
###################################################
# # Plot Data
# ## Plot before transform - cartesian
# fig, ax = plt.subplots()
# plt.suptitle('Cartesian plot')
# colors = ['red','blue','green']
# for d,c in zip(data_test,colors):    
#     ax.scatter(d[0],d[1],color=c)
# plt.show()

## Polar transform
data_polar_test = []
for d in data_test:
    rhos = []
    phis = []
    for i in range(len(d[0])):
        rho, phi = cart2pol(d[0][i],d[1][i])
        rhos.append(rho)
        phis.append(phi)
    cls = []
    cls.append(np.array(rhos))
    cls.append(np.array(phis))
    data_polar_test.append(np.array(cls))

data_polar_train = []
for d in data_train:
    rhos = []
    phis = []
    for i in range(len(d[0])):
        rho, phi = cart2pol(d[0][i],d[1][i])
        rhos.append(rho)
        phis.append(phi)
    cls = []
    cls.append(np.array(rhos))
    cls.append(np.array(phis))
    data_polar_train.append(np.array(cls))

# ## Plot Polar
# fig, ax = plt.subplots()
# plt.suptitle('Polar plot')
# colors = ['red','blue','green']
# for d,c in zip(data_polar_test,colors):    
#     ax.scatter(d[0],d[1],color=c)
# plt.show()


# Classify using r only
## Compute Scores using g_1b
test_scores = []
for k in range(len(data_polar_test)):
    for i in range(len(data_polar_test[k][0])):
        gi = []
        for j in range(len(data_polar_train)):
            gi.append(g_1b(np.array(data_polar_test[k])[0,i],data_polar_train[j][0],.33))
        test_scores.append(np.array(gi))

test_scores = np.array(test_scores)
#print_c('Scores for test samples:',test_scores)

## Classify
predictions = []
for ts in test_scores:
    predictions.append(classifier(ts))
predictions = np.array(predictions)

#print('Classification:',predictions)

## Build confusion matrix
### Build labels
labels = []
for i,d in enumerate(data_test):
    for x in d[0]:
        labels.append(i)
labels = np.array(labels)

### Build CM
cm = np.zeros((dim,dim))
good = 0
for pred,lab in zip(predictions,labels):
    cm[pred,lab] += 1

### Compute test score
cm_score = 0.0
for i in range(dim):
    cm_score += cm[i,i]
cm_score = round(cm_score / len(labels), 2)

print_c('Confusion Matrix:\nScore: '+str(cm_score),cm)
