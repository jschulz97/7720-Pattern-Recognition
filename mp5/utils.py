import numpy as np

# Manually compute mean
def mean(data):
    sum = 0.0
    for val in data:
        sum += val

    return sum/len(data)


# Manually compute mean multivariate
def mean_mv(data):
    means = []
    for d in data:
        sum = 0.0
        for val in d:
            sum += val
        means.append(sum/len(d))

    return means


# Manually compute variance
def var(data):
    m = mean(data)

    sum = 0.0
    for val in data:
        sum += np.power(val - m, 2)

    return sum/len(data)


# Manually Compute covariance matrix
def cov(data):
    n = len(data[0])
    d = len(data)
    covs = np.zeros((d,d))
    means = mean_mv(data)

    for x in range(d):
        for y in range(d):
            x_v = (data[x] - means[x])
            y_v = (data[y] - means[y])
            covs[x,y] = np.sum(x_v * y_v) / n

    return np.array(covs)


# Mahalanobis Distance
def mahala(x1, x2, cov):
    temp = np.dot((x1 - x2).T, np.linalg.inv(cov))
    return np.dot(temp, (x1 - x2))


# Question 1b discriminate fx from MP2 - multivariate
def g_1b_mv(x,data,prior):
    d  = len(data)
    mn = mean_mv(data)
    cv = cov(data)
    md = mahala(x,mn,cv)

    gi = (-1 * .5 * md) + (-1 * d/2 * np.log(2*np.pi)) + (-.5 * np.log(np.linalg.det(cv))) + np.log(prior)

    return gi


# Question 1b discriminate fx from MP2
def g_1b(x,data,prior):
    d  = len(data)
    mn = mean(data)
    va = var(data)
    md = mahala(x,mn,va)

    gi = (-1 * .5 * md) + (-1 * d/2 * np.log(2*np.pi)) + (-.5 * np.log(np.linalg.det(va))) + np.log(prior)

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


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


# Pretty print for matrices
def print_c(str,mats):
    print(str)
    for c in mats:
        print(c)
    print()