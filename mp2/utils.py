import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d


###################################################
# Covariances
###################################################

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

    return np.array(covs)

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
# Distance and Discriminant fxs - Question 1
###################################################

# Mahalanobis Distance
def mahala(x1, x2, cov):
    temp = np.dot((x1 - x2).T, np.linalg.inv(cov))
    return np.dot(temp, (x1 - x2))

# PDF
def pdf(x,data,cov):
    det = np.linalg.det(cov)
    temp = 1 / (np.power(2*np.pi , len(data)/2) * np.power(det , .5))
    return temp * np.exp((-.5 * mahala(x,mean(data),cov)))

# Question 1b discriminate fx
def g_1b(x,data,prior):
    d  = len(data)
    mn = mean(data)
    cv = cov(data)
    md = mahala(x,mn,cv)

    gi = (-1 * .5 * md) + (-1 * d/2 * np.log(2*np.pi)) + (-.5 * np.log(np.linalg.det(cv))) + np.log(prior)

    return gi


###################################################
# Generate Normal distributions with given
# mean and covariance matrix
###################################################

# Generate random samples using whitening transform
def generate_dist(u,cov,n):
    m1 = [8,2]
    cv1 = [[4.1,0],[0,2.8]]
    num_samples = 1000

    data1 = np.random.multivariate_normal([0,0],[[1,0],[0,1]],num_samples)
    data1 = np.reshape(data1,(2,num_samples))

    evals = evalues(cv1)
    evals_mat = np.array([[evals[0],0.0],[0.0,evals[1]]])
    evects = evectors(cv1,evals)

    Y = np.dot(evects.T , data1)
    evals_mat = np.sqrt(evals_mat)

    W = np.dot(evals_mat, Y)

    new = W

    new = new + np.reshape(m1,(2,1))

    m1 = mean(new)
    cv1 = cov(new,m1)

    print_c('new mean',m1)
    print_c('new cov',cv1)


    eva, eve = np.linalg.eig(cv1)
    evals = evalues(cv1)
    print_c('evals by hand:',evals)
    print_c('evals by np:',eva)

    print_c('evects by hand:',evectors(cv1,evals))
    print_c('evects by np:',eve)

    print(mean(data1))


###################################################
# Discriminant fxs - Question 2
###################################################

# Discriminant fx for case 2
def case2(x,data,prior):
    mn = mean(data)
    cv = cov(data)
    md = mahala(x,mn,cv)

    return (-.5 * md) + np.log(prior)

# Eqn for line, case 2
def case2_line(x,data1, data2):
    mn1 = mean(data1)
    mn2 = mean(data2)
    cv = cov(data1)
    w = np.dot( np.linalg.inv(cv) , (mn1 - mn2) )
    x0 = (.5 * (mn1 + mn2))
    return (-w[0] * (x - x0[0])) / w[1] + x0[1]

# Discriminant fx for case 3
def case3(x,data,prior):
    mn = mean(data)
    cv = cov(data)
    x = np.array(x)
    c_i = np.linalg.inv(cv)
    c_d = np.linalg.det(cv)
    bigw   = (-.5 * c_i)
    smallw = np.dot(c_i , mn) 
    w0     = (-.5 * np.dot(np.dot(mn.T , c_i), mn)) - (.5 * np.log(c_d)) + np.log(prior)
    g = np.dot(np.dot(x.T , bigw) , x) + np.dot(smallw.T , x) + w0 

    return g


# Posterior Probability
def post(x,datum,prior,data,priors):
    c = []
    for d in data:
        c.append(cov(d))
    sum = 0
    for cj,wj,dj in zip(c,priors,data):
        sum += pdf(x,dj,cj) * wj
        
    return (pdf(x,datum,cov(datum)) * prior) / sum


# Plot PDFs and Decision bound for case 2
def plot_case_2_2d(data1,data2):
    x_1 = np.arange(-5, 15, .5)
    pdf1,pdf2 = get_pdfs(data1,data2,x_1,x_1)

    # Plot hyperplane
    hp_y = []
    hp_x = np.arange(-5, 15, 1)
    for xi in hp_x:
        hp_y.append(case2_line(xi,data1,data2))

    ax = plt.axes()
    ax.set_xlim(-5,15)
    ax.set_ylim(-5,15)
    ax.scatter(data1[0],data1[1],color='blue')
    ax.scatter(data2[0],data2[1],color='red')
    ax.plot(hp_x,hp_y,color='black')
    plt.show()


# Plot PDFs and Decision bound for case 2
def plot_case_2_3d(data1,data2):
    x_1 = np.arange(-5, 15, .5)
    y_1 = np.arange(-5, 15, .5)
    x_2 = x_1
    y_2 = y_1

    pdf1,pdf2 = get_pdfs(data1,data2,x_1,y_1)

    x_1 = np.outer(x_1, np.ones(len(x_1)))
    x_2 = np.outer(x_2, np.ones(len(x_2)))
    y_1 = np.outer(np.ones(len(y_1)),y_1)
    y_2 = np.outer(np.ones(len(y_2)),y_2)

    # Plot hyperplane
    hp_y = []
    hp_x = np.arange(-5, 15, 1)
    for xi in hp_x:
        hp_y.append(case2_line(xi,data1,data2))
    hp_y = np.array(hp_y)

    # Plot post
    p_y = np.arange(-5, 15, 1)
    p_x = np.arange(-5, 15, 1)
    p_z1, p_z2 = [], []
    p_w1 = .8 
    p_w2 = .2
    for xi in p_x:
        temp1,temp2 = [],[]
        for yi in p_y:
            temp1.append(post([xi,yi], data1, p_w1, [data1,data2], [p_w1,p_w2]))
            temp2.append(post([xi,yi], data2, p_w2, [data1,data2], [p_w1,p_w2]))
        p_z1.append(temp1)
        p_z2.append(temp2)
    p_z1 = np.array(p_z1)
    p_z2 = np.array(p_z2)
    p_x = np.outer(p_x, np.ones(len(p_x)))
    p_y = np.outer(np.ones(len(p_y)),p_y)

    # Plot PDFs and Decision Bound
    ax = plt.axes(projection='3d')
    pdf1 = np.ma.array(pdf1, mask=np.isnan(pdf1))
    pdf2 = np.ma.array(pdf2, mask=np.isnan(pdf2))

    ax.set_zlim(0.0, .05)
    ax.plot_surface(x_1,y_1,pdf1,cmap='viridis',linewidth=0,antialiased=False,edgecolor='none',vmin=np.nanmin(pdf1), vmax=np.nanmax(pdf1))
    ax.plot_surface(x_2,y_2,pdf2,cmap='magma',linewidth=0,antialiased=False,edgecolor='none',vmin=np.nanmin(pdf2), vmax=np.nanmax(pdf2))
    ax.scatter3D(hp_x,hp_y,.01)
    plt.show()
    
    # Plot posterior
    ax = plt.axes(projection='3d')
    ax.plot_surface(p_x,p_y,p_z1,cmap='viridis',linewidth=0,antialiased=False,edgecolor='none')
    ax.plot_surface(p_x,p_y,p_z2,cmap='magma',linewidth=0,antialiased=False,edgecolor='none')
    plt.show()


# Plot PDFs and Decision bound for case 3
def plot_case_3_2d(data1,data2,prior1):
    x_1 = np.arange(-5, 15, .5)
    pdf1,pdf2 = get_pdfs(data1,data2,x_1,x_1)

    dp_1 = []
    # Plot Decision boundaries
    for d1_1 in x_1:
        temp = []
        for d1_2 in x_1:
            if(case3([d1_1,d1_2],data1,prior1) < .01 and case3([d1_1,d1_2],data2,1-prior1) < .01):
                temp.append([d1_1,d1_2])
        if(temp != []):
            dp_1.append(temp)
    dp_1 = np.array(dp_1)
    print_c('case 3 decision bound:',dp_1)
    #for d2_1, d2_2 in zip(data2[0],data2[1]):

    ax = plt.axes()
    ax.set_xlim(-5,15)
    ax.set_ylim(-5,15)
    ax.scatter(data1[0],data1[1],color='blue')
    ax.scatter(data2[0],data2[1],color='red')
    ax.plot(dp_1[:,0],dp_1[:,1],.1,color='black')
    plt.show()


# Plot PDFs and Decision bound for case 3
def plot_case_3_3d(data1,data2):
    x_1 = np.arange(-5, 15, .5)

    pdf1,pdf2 = get_pdfs(data1,data2,x_1,x_1)

    x_1 = np.outer(x_1, np.ones(40))
    x_2 = np.outer(x_2, np.ones(40))
    y_1 = np.outer(np.ones(40),y_1)
    y_2 = np.outer(np.ones(40),y_2)


    # Plot hyperplane
    hp_y = []
    hp_x = np.arange(-5, 15, .5)
    for xi in hp_x:
        hp_y.append(np.sum(case3(xi,[data1,data2],.5)))

    hp_y = np.array(hp_y)
    hp_y[hp_y > .0001] = np.NaN

    c = cov(data1)

    # Plot post
    p_y = []
    p_x = np.arange(-5, 15, 1)
    p_w1 = .8 
    for xi in p_x:
        temp = []
        for yi in p_x:
            temp.append(pdf([xi,yi],data1,c) * p_w1)
        p_y.append(temp)
    p_y = np.array(p_y)


    ax = plt.axes(projection='3d')
    pdf1 = np.ma.array(pdf1, mask=np.isnan(pdf1))
    pdf2 = np.ma.array(pdf2, mask=np.isnan(pdf2))

    ax.set_zlim(0.0, .05)
    ax.plot_surface(x_1,y_1,pdf1,cmap='viridis',linewidth=0,antialiased=False,edgecolor='none',vmin=np.nanmin(pdf1), vmax=np.nanmax(pdf1))
    ax.plot_surface(x_2,y_2,pdf2,cmap='magma',linewidth=0,antialiased=False,edgecolor='none',vmin=np.nanmin(pdf2), vmax=np.nanmax(pdf2))
    ax.scatter3D(hp_x,hp_y,.01)
    plt.show()



###################################################
# Helpful extras
###################################################

# Classifier
def classifier(scores):
    max = scores[0]
    cls = 0
    for i in range(len(scores)):
        if(scores[i] > max):
            cls = i
            max = scores[i]
    return cls

# Pretty print for matrices
def print_c(str,mats):
    print(str)
    for c in mats:
        print(c)
    print()

# Get PDFs
def get_pdfs(data1,data2,x,y):
    cov1 = cov(data1)
    cov2 = cov(data2)

    print('Sampling PDFs...')
    
    x_1, x_2 = x, x
    y_1, y_2 = y, y

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

    return pdf1,pdf2


# Double-check mean and covariances
def test_cov(data):
    # Calculate Mean
    means = []
    for c in data:
        means.append(mean(c))
    print_c('Means by hand:',means)

    # From numpy
    np_means = []
    for c in data:
        np_means.append([np.mean(c[0]),np.mean(c[1]),np.mean(c[2])])
    np_means = np.array(np_means)
    print_c('Means from np:',np_means)


    # Calculate covariance matrix
    covs = []
    for c,m in zip(data,means):
        covs.append(cov(c,m))
    print_c('Covs by hand:',covs)

    # From numpy
    np_cov = []
    for c in data:
        np_cov.append(np.cov(c))
    np_cov = np.array(np_cov)
    print_c('Covs from np:',np_cov)
