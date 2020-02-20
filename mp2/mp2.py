import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

from utils import *


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

plt.show()


###################################################
# Question 2b&c
###################################################
dim = 1000
data1 = np.random.multivariate_normal([8,2],[[4.1,0],[0,2.8]],dim).T
data2 = np.random.multivariate_normal([2,8],[[4.1,0],[0,2.8]],dim).T

plot_case_2(data1,data2)


###################################################
# Question 2d
###################################################
dim = 1000
data1 = np.random.multivariate_normal([8,2],[[4.1,0.4],[0.4,2.8]],dim).T
data2 = np.random.multivariate_normal([2,8],[[4.1,0.4],[0.4,2.8]],dim).T

plot_case_2(data1,data2)


###################################################
# Question 2e
###################################################
dim = 1000
data1 = np.random.multivariate_normal([8,2],[[2.1,1.5],[1.5,3.8]],dim).T
data2 = np.random.multivariate_normal([2,8],[[4.1,0.4],[0.4,2.8]],dim).T

plot_case_3(data1,data2)

