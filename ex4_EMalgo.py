import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, poisson

# Initialization of the parameters.
def init_params(X, S, K):
    #mean
    xmin = np.min(X)
    xmax = np.max(X)
    mu = np.random.uniform(xmin, xmax, (K, 2))

    #covariance
    sigma = np.zeros((K, 2, 2))
    for k in range(K):
        sigma[k] = np.diag(np.random.uniform(0, 1, 2))

    #Poisson parameter
    lbda = np.random.rand(K)

    #Mixing coef
    pi = np.random.rand(K)

    return mu, sigma, lbda, pi


#E-step of the EM Algorithm
def e_step(X, S, mu, sigma, lbda, pi):
    N, _ = X.shape
    gamma = np.zeros((N, K))

    for n in range(N): #compute the posterior probabilities
        denom_n = 0 #initialize the denominator for each n
        for k in range(K):
            gamma[n, k] = pi[k]*multivariate_normal(mu[k], sigma[k]).pdf(X[n]) \
            *poisson(lbda[k]).pmf(S[n])
            denom_n += gamma[n, k]

        for k in range(K):
            gamma[n, k] = gamma[n, k]/denom_n

    return gamma


#M-step of the EM Algorithm
def m_step(X, S, gamma):
    N, _ = X.shape

    Nktab = np.sum(gamma, axis=0)

    #Initialization of each new parameter
    mu = np.zeros((K, 2))
    sigma = np.zeros((K, 2, 2))
    lbda = np.zeros(K)
    pi = np.zeros(K)

    #mean
    for k in range(K):
        for n in range(N):
            mu[k] += gamma[n, k]*X[n]
        mu[k] = mu[k] / Nktab[k]

    #covariance
    for k in range(K):
        for n in range(N):
            sigma[k] += gamma[n, k] * np.outer((X[n] - mu[k]), (X[n] - mu[k]))
        sigma[k] = np.diag(np.diag(sigma[k] / Nktab[k]))

    #Poisson param
    for k in range(K):
        for n in range(N):
            lbda[k] += gamma[n, k] * S[n]
        lbda[k] = lbda[k] / Nktab[k]

    #Mixing coef
    for k in range(K):
        pi[k] = Nktab[k]/N

    return mu, sigma, lbda, pi


#Compute the log likelihood (used for convergence criterion)
def loglikelihood(X, mu, sigma, lbda, pi):
    l = 0
    N, _ = X.shape
    for n in range(N):
        arg_ln = 0
        for k in range(K):
            arg_ln += pi[k]*multivariate_normal(mu[k], sigma[k]).pdf(X[n])\
                 *poisson(lbda[k]).pmf(S[n])
        l += np.log(arg_ln)
    return l[0]


#Find the color of the point according to the posterior probabilities
def find_color(gamma, colors, n):
    c = 0
    for k in range(K):
        c += int(gamma[n, k]*colors[k])
    str_hexa = str(hex(c))[2:]
    return "#"+"0"*(6-len(str_hexa))+str_hexa


#Display the result obtained at a given step
def disp_res(X, mu, sigma, lbda, gamma, colors, K, step):
    N, _ = X.shape
    plt.figure()

    if step == 1: #Before the first M-step
        plt.scatter(X[:, 0], X[:, 1], c='black', s=S)
        for k in range(K):
            rv = multivariate_normal(mu[k], sigma[k])
            color = "#" + "0"*(6-len(str(hex(colors[k]))[2:])) + str(hex(colors[k]))[2:]
            plt.contour(x, y, rv.pdf(pos), colors=color, linewidths=1, alpha=1)
    else: #any step except the first one
        # Plotting the data points
        for n in range(N):
            c = find_color(gamma, colors, n)
            plt.scatter(X[n, 0], X[n, 1], c=c, s=S[n]*3)

        #Ploot the cluster Gaussians
        for k in range(K):
            plt.scatter(mu[k, 0], mu[k, 1], c=colors[k], marker='x', s=50)
            rv = multivariate_normal(mu[k], sigma[k])
            color = "#"+"0"*(6-len(str(hex(colors[k]))[2:]))+str(hex(colors[k]))[2:]
            plt.contour(x, y, rv.pdf(pos), colors=color, linewidths=lbda[k]/4, alpha=0.4)
    plt.title(f'K={K}, step={step}')
    plt.show()


#Load the data
x_path = "./X.txt"
s_path = "./S.txt"
X = pd.read_csv(x_path, sep=" ", header=None).to_numpy()
S = pd.read_csv(s_path, sep=" ", header=None).to_numpy()


#Preparing the plots
x = np.linspace(-3, 10, 200)
y = np.linspace(-3, 10, 200)

xx, yy = np.meshgrid(x, y)
pos = np.dstack((xx, yy))


#Colors used in find_color() function
red = 0xff0000
blue = 0x00ccff
green = 0xadff2f
yellow = 0xffff00
pink = 0xff66cc
colors = [pink, blue, green, yellow]

#Parameters of the EM Algorithm
np.random.seed(0)
max_iter = 10
eps = 1e-5
K = 3

#Start of EM ALgorithm
mu, sigma, lbda, pi = init_params(X, S, K) #Initialization

step = 1

saving_path = f'./img/ex4/K=3/test3/'
disp_res(X, mu, sigma, lbda, [], colors, K, step)
plt.savefig(saving_path+f'K={K}, step={step}')

loglkhOld = 0
loglkhNew = np.inf

#EM loop
# while np.abs(loglkhNew - loglkhOld) > eps and step < max_iter:
while np.abs(loglkhNew - loglkhOld) > eps: #Convergence criterion

    step += 1

    gamma = e_step(X, S, mu, sigma, lbda, pi) #E_step
    mu, sigma, lbda, pi = m_step(X, S, gamma) #M-step

    if step%2 == 0:
        disp_res(X, mu, sigma, lbda, gamma, colors, K, step)
        plt.savefig(saving_path+f'K={K}, step={step}')
    loglkhOld = loglkhNew
    #Compute new log likelihood
    loglkhNew = loglikelihood(X, mu, sigma, lbda, pi)
    print(f'step={step}, likelihood={loglkhNew}')


