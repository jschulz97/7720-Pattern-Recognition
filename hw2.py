from matplotlib import pyplot as plt
import numpy as np

class N:
    def __init__(self, u, s, P):
        self.u = u
        self.s = s
        self.P = P

    def __call__(self,x):
        return self.P * np.exp(-1 * (np.power(x - self.u, 2)) / (2 * np.power(self.s, 2))) / np.power(2 * np.pi * np.power(self.s, 2) , .5)


class cplusone:
    def __init__(self,n,l):
        self.n = n
        self.l = l
    
    def __call__(self,x):
        n_sum = sum([ni(x) for ni in self.n])
        return (1 - self.l) * n_sum


x = np.arange(-5, 5, .01)

exp = 2
if(exp == 1):
    P1 = .5
    n1 = N(1,1,P1)
    y1 = []

    P2 = .5
    n2 = N(-1,1,P2)
    y2 = []

    c  = cplusone([n1,n2],.25)
    cy = []

    # c = []
    # for i in np.arange(0,1.1,.1):
    #     c.append(cplusone([n1,n2],i))

    fig = plt.figure()
    # for ci in c:
    #     y = []
    #     for i in x:
    #         y.append(ci(i))
    #     plt.plot(x,y)

    #sum = 0.0
    for i in x:
        #sum += n1(i)
        y1.append(n1(i))
        y2.append(n2(i))
        cy.append(c(i))

    #print(sum/len(ls))

    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.plot(x,cy)
    plt.show()

if(exp == 2):
    P1 = .33
    n1 = N(1,1,P1)
    y1 = []

    P2 = .67
    n2 = N(0,.25,P2)
    y2 = []

    c  = cplusone([n1,n2],.5)
    cy = []

    fig = plt.figure()

    for i in x:
        y1.append(n1(i))
        y2.append(n2(i))
        cy.append(c(i))

    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.plot(x,cy)
    plt.show()

if(exp == 3):
    n1 = N(1,1,1)
    y1, y2 = [], []
    n2 = N(1,.5,1)

    for i in x:
        y1.append(n1(i))
        y2.append(n2(i))

    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.show()