import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

# Pretty print for matrices
def print_c(str,mats):
    print(str)
    for c in mats:
        print(c)
    print()


# Manually compute mean
def mean(data):
    means = []

    for d in data:
        sum = 0
        for val in d:
            sum += val
        means.append(sum/len(d))

    means = np.array(means)
    return means


# Manually Compute covariance matrix
def cov(data,means):
    n = len(data[0])
    d = len(data)
    covs = np.zeros((d,d))

    for x in range(d):
        for y in range(d):
            x_v = (data[x] - means[x]) / n
            y_v = (data[y] - means[y]) / n
            covs[x,y] = np.sum(x_v * y_v)

    return covs


# Mahalanobis Distance
def mahala(x1, x2, cov):
    temp = np.dot((x1 - x2).T, np.linalg.inv(cov))
    return np.dot(temp, (x1 - x2))


# Question 1b discriminate fx
def g_1b(x,data,prior):
    d  = len(data)
    mn = mean(data)
    cv = cov(data,mn)
    md = mahala(x,mn,cv)

    gi = (-1 * .5 * md) + (-1 * d/2 * np.log(2*np.pi)) + (-.5 * np.log(np.linalg.det(cv))) + np.log(prior)

    return gi


# Classifier
def classifier(scores):
    max = scores[0]
    cls = 0
    for i in range(len(scores)):
        if(scores[i] > max):
            cls = i
            max = scores[i]
    return cls


# PDF
def pdf(x,data,cov):
    det = np.linalg.det(cov)
    temp = 1 / (np.power(2*np.pi , len(x)/2) * np.power(det , .5))
    return temp * np.exp((-.5 * mahala(x,mean(data),cov)))


# Eigenvalues
def evalues(cov):
    b = (-1 * cov[0][0]) - cov[1][1]
    c = (-1 * cov[0][1] * cov[1][0]) + (cov[0][0] * cov[1][1])

    l1 = .5 * (-1 * b + pow(b*b - (4 * c),.5))
    l2 = .5 * (-1 * b - pow(b*b - (4 * c),.5))

    return np.array([l1,l2])


# Eigenvectors
def evectors(cov,evals):
    ev01 = 1
    ev02 = cov[0][1] / (evals[0] - cov[0][0] + 0.00000001)

    ev11 = cov[0][1] / (evals[1] - cov[0][0] + 0.00000001)
    ev12 = 1

    return np.array([[ev01,ev02],[ev11,ev12]])


###################################################
# Load Data
###################################################
data = loadmat('data_class3.mat')
data = data['Data'][0]

# for c in data:
#     print_c('',c)
# input()

class_1 = np.zeros((3,10))
class_2 = np.zeros((3,10))
class_3 = np.zeros((3,10))

class_1[0] = data[0][0]
class_1[1] = data[0][1]
class_1[2] = data[0][2]
class_2[0] = data[1][0]
class_2[1] = data[1][1]
class_2[2] = data[1][2]
class_3[0] = data[2][0]
class_3[1] = data[2][1]
class_3[2] = data[2][2]

dataset = [class_1,class_2,class_3]


###################################################
# Question 1
# Calculate Membership using Discriminant
###################################################
test_samples = [
    [1,3,2],
    [4,6,1],
    [7,-1,0],
    [-2,6,5]
]
test_scores = []
priors = [.6,.2,.2]

# Compute Scores using g_1b
for ts in test_samples:
    gi = []
    for c,p in zip(dataset,priors):
        gi.append(g_1b(ts,c,p))
    test_scores.append(np.array(gi))

test_scores = np.array(test_scores)
print_c('Scores for test samples:',test_scores)

# Classify
classes = []
for ts in test_scores:
    classes.append(classifier(ts))

print('Classification:',classes)


###################################################
# Plot Question 1
###################################################
colors = ['blue','green','orange']
ax = plt.axes(projection='3d')
for i,c in enumerate(data):    
    ax.scatter3D(c[0],c[1],c[2],color=colors[i])
    #plt.quiver(means[i][0], means[i][1], evectors[i][0][0], evectors[i][0][1], angles='xy', scale_units='xy', scale=1, headlength=3, headwidth=3, headaxislength=3)
    #plt.quiver(means[i][0], means[i][1], evectors[i][1][0], evectors[i][1][1], angles='xy', scale_units='xy', scale=1, headlength=3, headwidth=3, headaxislength=3)

colors = ['magenta','yellow','cyan','red']
for i,c in enumerate(test_samples): 
    ax.scatter3D(c[0],c[1],c[2],color=colors[i])

#plt.show()



###################################################
# Question 2
###################################################
data1 = np.random.multivariate_normal([8,2],[[4.1,0],[0,2.8]],1000).T
data2 = np.random.multivariate_normal([2,8],[[4.1,0],[0,2.8]],1000).T

m1 = mean(data1)
m2 = mean(data2)
print(m1, m2)
print(cov(data1,m1))
print(cov(data2,m2))

print(np.cov(data1))


# Generate random samples
# m1 = [8,2]
# cv1 = [[4.1,0],[0,2.8]]
# num_samples = 1000

# data1 = np.random.multivariate_normal([0,0],[[1,0],[0,1]],num_samples)
# data1 = np.reshape(data1,(2,num_samples))

# evals = evalues(cv1)
# evals_mat = np.array([[evals[0],0.0],[0.0,evals[1]]])
# evects = evectors(cv1,evals)

# Y = np.dot(evects.T , data1)
# evals_mat = np.sqrt(evals_mat)

# W = np.dot(evals_mat, Y)

# new = W

# new = new + np.reshape(m1,(2,1))

# m1 = mean(new)
# cv1 = cov(new,m1)

# print_c('new mean',m1)
# print_c('new cov',cv1)


# eva, eve = np.linalg.eig(cv1)
# evals = evalues(cv1)
# print_c('evals by hand:',evals)
# print_c('evals by np:',eva)

# print_c('evects by hand:',evectors(cv1,evals))
# print_c('evects by np:',eve)

#print(mean(data1))










###################################################
# Calculate Covariances
###################################################

# # Calculate Mean
# means = []
# for c in data:
#     means.append(mean(c))
# print_c('Means by hand:',means)

# # From numpy
# np_means = []
# for c in data:
#     np_means.append([np.mean(c[0]),np.mean(c[1]),np.mean(c[2])])
# np_means = np.array(np_means)
# print_c('Means from np:',np_means)


# # Calculate covariance matrix
# covs = []
# for c,m in zip(data,means):
#     covs.append(cov(c,m))
# print_c('Covs by hand:',covs)

# # From numpy
# np_cov = []
# for c in data:
#     np_cov.append(np.cov(c))
# np_cov = np.array(np_cov)
# print_c('Covs from np:',np_cov)


