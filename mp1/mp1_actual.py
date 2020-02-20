import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt

# Pretty print for matrices
def print_c(str,mats):
    print(str)
    for c in mats:
        print(c)
    print()


###################################################
# Load Data
###################################################
data = loadmat('data_class4.mat')
data = data['Data'][0]

class_1 = np.zeros((2,400))
class_2 = np.zeros((2,400))
class_3 = np.zeros((2,600))
class_4 = np.zeros((2,400))

class_1[0] = data[0][0]
class_1[1] = data[0][1]
class_2[0] = data[1][0]
class_2[1] = data[1][1]
class_3[0] = data[2][0]
class_3[1] = data[2][1]
class_4[0] = data[3][0]
class_4[1] = data[3][1]

data = [class_1,class_2,class_3,class_4]


###################################################
# Calculate Covariances
###################################################

# From numpy
np_means = []
for c in data:
    np_means.append([np.mean(c[0]),np.mean(c[1])])
np_means = np.array(np_means)
print_c('Means from np:',np_means)

# Manually compute mean
means = []
for c in data:
    sum_0 = 0
    sum_1 = 0
    for val in c[0]:
        sum_0 += val
    for val in c[1]:
        sum_1 += val
    means.append([sum_0 / len(c[0]), sum_1 / len(c[0])])

means = np.array(means)
print_c('Means by hand:',means)

# From numpy
np_cov = []
for c in data:
    np_cov.append(np.cov(c[0],c[1]))
np_cov = np.array(np_cov)
print_c('Covs from np:',np_cov)

# Manually calculate covariance
covs = []
for c,m in zip(data,means):  
    n = len(c[0])
    sum = 0
    for x,y in zip(c[0],c[1]):
        sum += ((x-m[0]) * (x-m[0]))
    xx = sum / n

    sum = 0
    for x,y in zip(c[0],c[1]):
        sum += ((y-m[1]) * (y-m[1]))
    yy = sum / n

    sum = 0
    for x,y in zip(c[0],c[1]):
        sum += ((x-m[0]) * (y-m[1]))
    xy = sum / n

    covs.append([[xx,xy],[xy,yy]])

covs = np.array(covs)
print_c('Covs by hand:',covs)


###################################################
# Compute Eigenvectors/Eigenvalues
###################################################

evalues  = []
evectors = []
for cov in covs:
    # Eigenvalues
    b = (-1 * cov[0][0]) - cov[1][1]
    c = (-1 * cov[0][1] * cov[1][0]) + (cov[0][0] * cov[1][1])

    l1 = .5 * (-1 * b - pow(b*b - (4 * c),.5))
    l2 = .5 * (-1 * b + pow(b*b - (4 * c),.5))

    evalues.append([l1,l2])

    # Eigenvector 1 - solve system
    ev01 = cov[0][1] / (l1 - cov[0][0])
    ev02 = 1

    # Normalize to eigenvalue
    c2a = evalues[-1][0]*evalues[-1][0] + evalues[-1][1]*evalues[-1][1]
    c2b = ev01*ev01 + ev02*ev02
    ca  = pow(c2a,.5)
    cb  = pow(c2b,.5)
    p = ca/cb
    ev01 = ev01 * p
    ev02 = ev02 * p
    # # Normalize to 1
    # c2 = ev01*ev01 + ev02*ev02
    # c  = pow(c2,.5)
    # ev01 = ev01 / c
    # ev02 = ev02 / c

    # Eigenvector 2 - solve system
    ev11 = cov[0][1] / (l2 - cov[0][0])
    ev12 = 1

    # Normalize to eigenvalue
    c2a = evalues[-1][0]*evalues[-1][0] + evalues[-1][1]*evalues[-1][1]
    c2b = ev11*ev11 + ev12*ev12
    ca  = pow(c2a,.5)
    cb  = pow(c2b,.5)
    p = ca/cb
    ev11 = ev11 * p
    ev12 = ev12 * p
    # # Normalize to 1
    # c2 = ev11*ev11 + ev12*ev12
    # c  = pow(c2,.5)
    # ev11 = ev11 / c
    # ev12 = ev12 / c
    
    evectors.append([[ev01,ev02],[ev11,ev12]])

evalues = np.array(evalues)
evectors = np.array(evectors)
print_c('Eigenvalues:',evalues)
print_c('Eigenvectors:',evectors)

# actualeva = []
# actualeve = []
# for d in covs:
#     eva, eve = np.linalg.eig(d)
#     actualeva.append(eva)
#     actualeve.append(eve)

# print_c('values:',actualeva)
# print_c('vectors:',actualeve)


###################################################
# Plot
###################################################
ax = plt.axes()
for i,c in enumerate(data):    
    ax.scatter(c[0],c[1])
    plt.quiver(means[i][0], means[i][1], evectors[i][0][0], evectors[i][0][1], angles='xy', scale_units='xy', scale=1, headlength=3, headwidth=3, headaxislength=3)
    plt.quiver(means[i][0], means[i][1], evectors[i][1][0], evectors[i][1][1], angles='xy', scale_units='xy', scale=1, headlength=3, headwidth=3, headaxislength=3)

plt.show()


