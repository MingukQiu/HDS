import numpy as np
import scipy as sp
from scipy import linalg as LA 
import matplotlib.pyplot as plt

np.random.seed(2022)

n = 100
Sigma = np.random.normal(0, 1, (n, n))
Sigma = (Sigma+Sigma.T)/2
lam,vec = LA.eigh(Sigma)
idx = lam.argsort()[::-1]
U = vec[:,idx]
theta_star = U[:,0]
lam = lam[idx]
nu = lam[0]-lam[1] #eigengap of Sigma

P = np.random.normal(0, 1, (n, n))
P = 0.9*(nu/2)*P/LA.norm(P,2)

Sigma_hat = Sigma + P 
lam,vec = LA.eigh(Sigma_hat)
idx = lam.argsort()[::-1]
theta_hat = vec[:,idx[0]]

P_tilde = U.T @ P @U
LHS = LA.norm(theta_hat-theta_star)

p_tilde = P_tilde[1:(n-1),0]

RHS = 2*LA.norm(p_tilde)/(nu-2*LA.norm(P,2))

print("LHS:", LHS, "RHS:", RHS)

def Psi(Delta, P, theta):
    y = np.inner(Delta, P@Delta)+2*np.inner(Delta,P@theta)
    return y 
LHS = nu*(1-np.inner(theta_hat,theta_star)**2)
Delta_hat = theta_hat - theta_star
RHS = Psi(Delta_hat, P, theta_star)

print("LHS:", LHS, "RHS:", RHS)

n = 500
d = int(0.2*n)
loop_num = 1000
nu_seq = np.arange(0.75,5,0.25)
df = np.zeros([loop_num,len(nu_seq)])
theta =  np.random.normal(0, 1, (d, 1))
theta = theta/LA.norm(theta)
nu_index = -1
for nu in nu_seq:
    nu_index = nu_index + 1
    mean = np.zeros(d)
    Sigma = nu*theta@theta.T + np.identity(d)
    for t in np.arange(0,loop_num,1,dtype=int):
        X = np.random.multivariate_normal(mean, Sigma, n) # n * d
        Sigma_hat = X.T@X/n
        lam,theta_hat = LA.eigh(Sigma_hat, eigvals=[d-1,d-1])
        df[t,nu_index] = LA.norm(theta_hat-theta) 
        print("nu:", nu, "loop:", t)

error = np.mean(df, axis = 0)

plt.plot(nu_seq,error)
plt.show()

