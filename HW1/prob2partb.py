# EECE5644 HW 1
# Jarrod Homer

import matplotlib.pyplot as plt # For general plotting
from matplotlib import cm

import numpy as np

from scipy.stats import multivariate_normal # MVN not univariate

np.set_printoptions(suppress=True)

# Set seed to generate reproducible "pseudo-randomness" (handles scipy's "randomness" too)
np.random.seed(7)

plt.rc('font', size=22)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)    # legend fontsize
plt.rc('figure', titlesize=22)  # fontsize of the figure title

# Number of samples to draw from each distribution
N = 10000

# Likelihood of each distribution to be selected AND class priors!!!
priors = np.array([0.2, 0.25, 0.25, 0.3])  
# Determine number of classes/mixture components
C = len(priors)
mu = np.array([[1, 1],
               [5, 5],
               [8, 8],
               [11, 11]])  # Gaussian distributions means

Sigma = np.array([[[6, 0],
                   [0, 6]],
                  [[20, 0],
                   [0, 20]],
                  [[6, 0],
                   [0, 6]],
                  [[15, 0],
                   [0, 15]]])  # Gaussian distributions covariance matrices

# Determine dimensionality from mixture PDF parameters
n = mu.shape[1]
# Output samples and labels
X = np.zeros([N, n])
y = np.zeros(N)

# Decide randomly which samples will come from each component
u = np.random.rand(N)
thresholds = np.cumsum(priors)

for c in range(C):
    c_ind = np.argwhere(u <= thresholds[c])[:, 0]  # Get randomly sampled indices for this component
    c_N = len(c_ind)  # No. of samples in this component
    y[c_ind] = c * np.ones(c_N)
    u[c_ind] = 1.1 * np.ones(c_N)  # Multiply by 1.1 to fail <= thresholds and thus not reuse samples
    X[c_ind, :] =  multivariate_normal.rvs(mu[c], Sigma[c], c_N)

# Plot the original data and their true labels
plt.figure(figsize=(12, 10))
plt.plot(X[y==0, 0], X[y==0, 1], 'b.', label="Class 1")
plt.plot(X[y==1, 0], X[y==1, 1], 'ro', label="Class 2")
plt.plot(X[y==2, 0], X[y==2, 1], 'g^', label="Class 3")
plt.plot(X[y==3, 0], X[y==3, 1], 'ms', label="Class 4")
plt.legend()
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("Data and True Labels")
plt.tight_layout()
plt.show()

Y = np.array(range(C))  # 0-(C-1)

# We are going to use a 0-1 loss matrix for th1is problem
Lambda = np.array([[0, 1, 2, 3],
                   [1, 0, 1, 2],
                   [2, 1, 0, 1],
                   [3, 2, 1, 0]])

print(Lambda)

# Calculate class-conditional likelihoods p(x|Y=j) for each label of the N observations
class_cond_likelihoods = np.array([multivariate_normal.pdf(X, mu[j], Sigma[j]) for j in Y])
class_priors = np.diag(priors)
print(class_cond_likelihoods.shape)
print(class_priors.shape)
class_posteriors = class_priors.dot(class_cond_likelihoods)
print(class_posteriors)

# We want to create the risk matrix of size 3 x N 
cond_risk = Lambda.dot(class_posteriors)
print(cond_risk)

# Get the decision for each column in risk_mat
decisions = np.argmin(cond_risk, axis=0)
print(decisions.shape)

# Plot for decisions vs true labels
fig = plt.figure(figsize=(12, 10))
marker_shapes = '.o^s' # Accomodates up to C=5
marker_colors = 'brgmy'

# Get sample class counts
sample_class_counts = np.array([sum(y == j) for j in Y])

# Confusion matrix
conf_mat = np.zeros((C, C))
for i in Y: # Each decision option
    for j in Y: # Each class label
        ind_ij = np.argwhere((decisions==i) & (y==j))
        conf_mat[i, j] = len(ind_ij)/sample_class_counts[j] # Average over class sample count

        # True label = Marker shape; Decision = Marker Color
        marker = marker_shapes[j] + marker_colors[i]
        plt.plot(X[ind_ij, 0], X[ind_ij, 1], marker_shapes[j]+'g')

        if i != j:
            plt.plot(X[ind_ij, 0], X[ind_ij, 1], marker_shapes[j]+'r')
            
print("Confusion matrix:")
print(conf_mat)

print("Minimum Probability of Error:")
prob_error = 1 - np.diag(conf_mat).dot(sample_class_counts / N)
print(prob_error)

plt.title("Minimum Probability of Error Classified Sampled Data:  {:.3f}".format(prob_error))
plt.show()