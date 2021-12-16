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

class_1 = np.zeros((2,10))
class_2 = np.zeros((2,10))
class_3 = np.zeros((2,10))

# class_1[0] = data[0][0]
# class_1[1] = data[0][1]
# class_1[2] = data[0][2]
# class_2[0] = data[1][0]
# class_2[1] = data[1][1]
# class_2[2] = data[1][2]
# class_3[0] = data[2][0]
# class_3[1] = data[2][1]
# class_3[2] = data[2][2]

# class_1[0] = np.linspace(-5,5,1000)
# class_2[0] = np.linspace(-5,5,1000)
# class_3[0] = np.linspace(-5,5,1000)

class_1 = np.random.multivariate_normal([0,0],[[1,0],[0,1]],100).T
class_2 = np.random.multivariate_normal([1,1],[[1,0],[0,1]],100).T
print(class_1.shape)
class_3 = np.random.multivariate_normal([.5,.5],[[1,0],[0,1]],50).T
temp = np.random.multivariate_normal([-.5,.5],[[1,0],[0,1]],50).T
class_3 = np.append(class_3,temp,axis=1)
print(class_3.shape)

dataset = [class_1,class_2,class_3]

###################################################
# Question 1
# Calculate Membership using Discriminant
###################################################
test_samples = [
    [.3,.3],
]
test_scores = []
priors = [.33,.33,.33]

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

input()

###################################################
# Plot Question 1
###################################################
colors = ['blue','green','orange']
ax = plt.axes(projection='3d')
for i,c in enumerate(data):    
    ax.scatter3D(c[0],c[1],c[2],color=colors[i])

colors = ['magenta','yellow','cyan','red']
for i,c in enumerate(test_samples): 
    ax.scatter3D(c[0],c[1],c[2],color=colors[i])

plt.show()


# ###################################################
# # Question 2b&c
# ###################################################
# dim = 1000
# data1 = generate_dist([8,2],[[4.1,0],[0,2.8]],dim)
# data2 = generate_dist([2,8],[[4.1,0],[0,2.8]],dim)

# plot_case_2_2d(data1,data2)
# plot_case_2_3d(data1,data2)


# ###################################################
# # Question 2d
# ###################################################
# dim = 1000
# data1 = generate_dist([8,2],[[4.1,0.4],[0.4,2.8]],dim).T
# data2 = generate_dist([2,8],[[4.1,0.4],[0.4,2.8]],dim).T

# plot_case_2_2d(data1,data2)
# plot_case_2_3d(data1,data2)


# ###################################################
# # Question 2e
# ###################################################
# dim = 1000
# data1 = generate_dist([8,2],[[2.1,1.5],[1.5,3.8]],dim).T
# data2 = generate_dist([2,8],[[4.1,0.4],[0.4,2.8]],dim).T

# plot_case_3_2d(data1,data2,.5)
# plot_case_3_3d(data1,data2,.5)
