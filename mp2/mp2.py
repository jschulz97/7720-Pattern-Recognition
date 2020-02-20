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
def cov(data):
    n = len(data[0])
    d = len(data)
    covs = np.zeros((d,d))
    means = mean(data)

    for x in range(d):
        for y in range(d):
            x_v = (data[x] - means[x])
            y_v = (data[y] - means[y])
            covs[x,y] = np.sum(x_v * y_v) / n

    return covs


# Mahalanobis Distance
def mahala(x1, x2, cov):
    temp = np.dot((x1 - x2).T, np.linalg.inv(cov))
    return np.dot(temp, (x1 - x2))


# Question 1b discriminate fx
def g_1b(x,data,prior):
    d  = len(data)
    mn = mean(data)
    cv = cov(data)
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
    temp = 1 / (np.power(2*np.pi , len(data)/2) * np.power(det , .5))
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

plt.show()


# Discriminant fx for b and d
def case2(x,data,prior):
    mn = mean(data)
    cv = cov(data)
    md = mahala(x,mn,cv)

    return (-.5 * md) + np.log(prior)

def case2_line(x,data1, data2):
    mn1 = mean(data1)
    mn2 = mean(data2)
    cv = cov(data1)
    w = np.dot( np.linalg.inv(cv) , (mn1 - mn2) )
    x0 = (.5 * (mn1 + mn2))
    #return np.dot( w.T , (x - x0))
    return (-w[0] * (x - x0[0])) / w[1] + x0[1]

# Discriminant fx for e
def case3(x,data,prior):
    mn = []
    cv = []
    md = []
    for d in data:
        mn.append(mean(d))
        cv.append(cov(d))

    g = []
    for m,c in zip(mn,cv):
        c_i = np.linalg.inv(c)
        c_d = np.linalg.det(c)
        bigw   = (-.5 * c_i)
        smallw = np.dot(c_i , m) 
        w0     = (-.5 * np.dot(np.dot(m.T , c_i), m)) - (.5 * np.log(c_d)) + np.log(prior)
        g.append( np.dot(np.dot(x.T , bigw) , x) + np.dot(smallw.T , x) + w0 )

    return g


###################################################
# Question 2
###################################################
dim = 1000
data1 = np.random.multivariate_normal([8,2],[[4.1,0],[0,2.8]],dim).T
data2 = np.random.multivariate_normal([2,8],[[4.1,0],[0,2.8]],dim).T

cov1 = cov(data1)
cov2 = cov(data2)

print('Sampling PDFs...')
# x_1, y_1 = data1[0], data1[1]
# x_2, y_2 = data2[0], data2[1]
x_1 = np.arange(-5, 15, .5)
x_2 , y_1, y_2 = x_1 , x_1, x_1
pdf1 = []
pdf2 = []
# for x1,x2 in zip(x_1,y_1):
#     pdf1.append(pdf([x1,x2],data1,cov1))

# for x1,x2 in zip(x_2,y_2):
#     pdf2.append(pdf([x1,x2],data2,cov2))

for x1 in x_1:
    temp = []
    for x2 in y_1:
        temp.append(pdf([x1,x2],data1,cov1))
    pdf1.append(temp)

for x1 in x_2:
    temp = []
    for x2 in y_2:
        temp.append(pdf([x1,x2],data2,cov2))
    pdf2.append(temp)

pdf1 = np.array(pdf1)
pdf2 = np.array(pdf2)

# clip data which is below 0.0
pdf1[pdf1 < 0.0001] = np.NaN
pdf2[pdf2 < 0.0001] = np.NaN

print('Done.')

x_1 = np.outer(x_1, np.ones(40))
x_2 = np.outer(x_2, np.ones(40))
y_1 = np.outer(np.ones(40),y_1)
y_2 = np.outer(np.ones(40),y_2)


# Plot hyperplane
hp_y = []
hp_x = np.arange(-5, 15, 1)
for xi in hp_x:
    hp_y.append(case2_line(xi,data1,data2))


ax = plt.axes(projection='3d')
pdf1 = np.ma.array(pdf1, mask=np.isnan(pdf1))
pdf2 = np.ma.array(pdf2, mask=np.isnan(pdf2))

ax.set_zlim(0.0, .05)
# ax.plot_trisurf(x_1,y_1,pdf1,cmap='viridis',linewidth=0,antialiased=False,edgecolor='none')
# ax.plot_trisurf(x_2,y_2,pdf2,cmap='magma',linewidth=0,antialiased=False,edgecolor='none')
ax.plot_surface(x_1,y_1,pdf1,cmap='viridis',linewidth=0,antialiased=False,edgecolor='none',vmin=np.nanmin(pdf1), vmax=np.nanmax(pdf1))
ax.plot_surface(x_2,y_2,pdf2,cmap='magma',linewidth=0,antialiased=False,edgecolor='none',vmin=np.nanmin(pdf2), vmax=np.nanmax(pdf2))
ax.scatter3D(hp_x,hp_y,.01)
# cmap = plt.cm.jet
# cmap.set_bad(alpha=0.0)
plt.show()


###################################################
# Question 2d
###################################################
dim = 1000
data1 = np.random.multivariate_normal([8,2],[[4.1,0.4],[0.4,2.8]],dim).T
data2 = np.random.multivariate_normal([2,8],[[4.1,0.4],[0.4,2.8]],dim).T

cov1 = cov(data1)
cov2 = cov(data2)

print('Sampling PDFs...')
x_1 = np.arange(-5, 15, .5)
x_2 , y_1, y_2 = x_1 , x_1, x_1
pdf1 = []
pdf2 = []

for x1 in x_1:
    temp = []
    for x2 in y_1:
        temp.append(pdf([x1,x2],data1,cov1))
    pdf1.append(temp)

for x1 in x_2:
    temp = []
    for x2 in y_2:
        temp.append(pdf([x1,x2],data2,cov2))
    pdf2.append(temp)

pdf1 = np.array(pdf1)
pdf2 = np.array(pdf2)

# clip data which is below 0.0
pdf1[pdf1 < 0.0001] = np.NaN
pdf2[pdf2 < 0.0001] = np.NaN

print('Done.')

x_1 = np.outer(x_1, np.ones(40))
x_2 = np.outer(x_2, np.ones(40))
y_1 = np.outer(np.ones(40),y_1)
y_2 = np.outer(np.ones(40),y_2)


# Plot hyperplane
hp_y = []
hp_x = np.arange(-5, 15, 1)
for xi in hp_x:
    hp_y.append(case2_line(xi,data1,data2))


ax = plt.axes(projection='3d')
pdf1 = np.ma.array(pdf1, mask=np.isnan(pdf1))
pdf2 = np.ma.array(pdf2, mask=np.isnan(pdf2))

ax.set_zlim(0.0, .05)
ax.plot_surface(x_1,y_1,pdf1,cmap='viridis',linewidth=0,antialiased=False,edgecolor='none',vmin=np.nanmin(pdf1), vmax=np.nanmax(pdf1))
ax.plot_surface(x_2,y_2,pdf2,cmap='magma',linewidth=0,antialiased=False,edgecolor='none',vmin=np.nanmin(pdf2), vmax=np.nanmax(pdf2))
ax.scatter3D(hp_x,hp_y,.01)
plt.show()



###################################################
# Question 2e
###################################################
dim = 1000
data1 = np.random.multivariate_normal([8,2],[[2.1,1.5],[1.5,3.8]],dim).T
data2 = np.random.multivariate_normal([2,8],[[4.1,0.4],[0.4,2.8]],dim).T

cov1 = cov(data1)
cov2 = cov(data2)

print('Sampling PDFs...')
x_1 = np.arange(-5, 15, .5)
x_2 , y_1, y_2 = x_1 , x_1, x_1
pdf1 = []
pdf2 = []

for x1 in x_1:
    temp = []
    for x2 in y_1:
        temp.append(pdf([x1,x2],data1,cov1))
    pdf1.append(temp)

for x1 in x_2:
    temp = []
    for x2 in y_2:
        temp.append(pdf([x1,x2],data2,cov2))
    pdf2.append(temp)

pdf1 = np.array(pdf1)
pdf2 = np.array(pdf2)

# clip data which is below 0.0
pdf1[pdf1 < 0.0001] = np.NaN
pdf2[pdf2 < 0.0001] = np.NaN

print('Done.')

x_1 = np.outer(x_1, np.ones(40))
x_2 = np.outer(x_2, np.ones(40))
y_1 = np.outer(np.ones(40),y_1)
y_2 = np.outer(np.ones(40),y_2)


# Plot hyperplane
hp_y = []
hp_x = np.arange(-5, 15, .5)
for xi in hp_x:
    hp_y.append(np.sum(case3(xi,[data1,data2],1)))

hp_y = np.array(hp_y)
hp_y[hp_y > .0001] = np.NaN


ax = plt.axes(projection='3d')
pdf1 = np.ma.array(pdf1, mask=np.isnan(pdf1))
pdf2 = np.ma.array(pdf2, mask=np.isnan(pdf2))

ax.set_zlim(0.0, .05)
ax.plot_surface(x_1,y_1,pdf1,cmap='viridis',linewidth=0,antialiased=False,edgecolor='none',vmin=np.nanmin(pdf1), vmax=np.nanmax(pdf1))
ax.plot_surface(x_2,y_2,pdf2,cmap='magma',linewidth=0,antialiased=False,edgecolor='none',vmin=np.nanmin(pdf2), vmax=np.nanmax(pdf2))
ax.scatter3D(hp_x,hp_y,.01)
plt.show()




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


