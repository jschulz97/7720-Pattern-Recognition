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
# print_c('Scores for test samples:',test_scores)


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

# Plot Data
## Plot before transform - cartesian
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
ax1.set_title('Train - Cartesian')
ax2.set_title('Test - Cartesian')
ax3.set_title('Train - Polar')
ax4.set_title('Test - Polar')
colors = ['red','blue','green']
for d,c in zip(data_train,colors):    
    ax1.scatter(d[0],d[1],color=c)
for d,c in zip(data_test,colors):    
    ax2.scatter(d[0],d[1],color=c)
for d,c in zip(data_polar_train,colors):    
    ax3.scatter(d[0],d[1],color=c)
for d,c in zip(data_polar_test,colors):    
    ax4.scatter(d[0],d[1],color=c)
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
# print_c('Scores for test samples:',test_scores)

## Classify
predictions = []
for ts in test_scores:
    predictions.append(classifier(ts))
predictions = np.array(predictions)

# print('Classification:',predictions)

## Build confusion matrix
### Build labels
labels = []
for i,d in enumerate(data_polar_test):
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



## Get mu_n and sigma_n
sigma   = .25
mu_0    = 0
sigma_0 = 100
mu_hat  = []
va_hat  = []
for d in data_polar_test:
    mu_hat.append(mean(d[0]))
for d in data_polar_test:
    va_hat.append(var(d[0]))
mu_hat  = np.array(mu_hat)
va_hat  = np.array(va_hat)
mu_n    = (dim * sigma_0 * mu_hat) / (dim * sigma_0 + sigma) + (sigma * mu_0) / (dim * sigma_0 + sigma)
sigma_n = (sigma_0 * sigma) / (dim * sigma_0 + sigma)
print_c("Bayesian Estimates\nMeans:",mu_n)
print('Sigma:\n'+str(sigma_n),'\n\n')

# Classify using r only - Bayesian parameter estimation
## Compute Scores using g_1b
test_scores = []
for k in range(len(data_polar_test)):
    for i in range(len(data_polar_test[k][0])):
        gi = []
        for j in range(len(data_polar_train)):
            gi.append(g_1b_bpe(np.array(data_polar_test[k])[0,i],mu_n[j],sigma+sigma_n,.33))
        test_scores.append(np.array(gi))

# test_scores = np.array(test_scores)
# print_c('Scores for test samples:',test_scores)

## Classify
predictions = []
for ts in test_scores:
    predictions.append(classifier(ts))
predictions = np.array(predictions)

# print('Classification:',predictions)

## Build confusion matrix
### Build labels
labels = []
for i,d in enumerate(data_polar_test):
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
