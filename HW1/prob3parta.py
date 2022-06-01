# EECE5644 HW 1
# Jarrod Homer

import matplotlib.pyplot as plt # For general plotting
from matplotlib import cm

import numpy as np

from scipy.stats import multivariate_normal # MVN not univariate

from sklearn.metrics import confusion_matrix

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

file = open("HW1/winequality-white.csv")
X = np.genfromtxt(file, delimiter=";")
X = X[1:,:]
N=len(X)
print(X.shape)

labels = X[:,11]
X = X[:,:11]
print(X.shape)


n=11
C=7
c=np.arange(3,10,1)
print(c)
mu=np.zeros((C,n))
Sigma=np.zeros((C,n,n))

priors=np.zeros(C)
con_lambda=0.01
for i in c:
    count = (labels == i).sum()
    priors[i-3]=count/N
    mu[i-3]=np.mean(X[labels==i,:], axis=0)
    Sigma[i-3]=np.cov(X[labels==i,:],rowvar=False)+con_lambda*np.identity(n)
print(mu)
#print(Sigma)

#print(priors)

###### Model

# Min prob. of error classifier
# Conditional likelihoods of each class given x, shape (C, N)
class_cond_likelihoods = np.array([multivariate_normal.pdf(X, mu[c,:], Sigma[c,:,:]) for c in range(C)])
# Take diag so we have (C, C) shape of priors with prior prob along diagonal
class_priors = np.diag(priors)
# class_priors*likelihood with diagonal matrix creates a matrix of posterior probabilities
# with each class as a row and N columns for samples, e.g. row 1: [p(y1)p(x1|y1), ..., p(y1)p(xN|y1)]
class_posteriors = class_priors.dot(class_cond_likelihoods)

# MAP rule, take largest class posterior per example as your decisions matrix (N, 1)
# Careful of indexing! Added np.ones(N) just for difference in starting from 0 in Python and labels={1,2,3}
decisions = np.argmax(class_posteriors, axis=0) + 3*np.ones(N) 

# Simply using sklearn confusion matrix
print("Confusion Matrix (rows: Predicted class, columns: True class):")
conf_mat = confusion_matrix(decisions, labels)
print(conf_mat)
correct_class_samples = np.sum(np.diag(conf_mat))
print("Total Mumber of Misclassified Samples: {:d}".format(N - correct_class_samples))

# Alternatively work out probability error based on incorrect decisions per class
# perror_per_class = np.array(((conf_mat[1,0]+conf_mat[2,0])/Nl[0], (conf_mat[0,1]+conf_mat[2,1])/Nl[1], (conf_mat[0,2]+conf_mat[1,2])/Nl[2]))
# prob_error = perror_per_class.dot(Nl.T / N)

prob_error = 1 - (correct_class_samples / N)
print("Empirically Estimated Probability of Error: {:.4f}".format(prob_error))

# Plot for decisions vs true labels
fig = plt.figure(figsize=(10, 10))

L=c
print(c)
marker_shapes = '.o^1s+*'
marker_colors = 'rbgmgbrrbgmgbr' 
for r in L: # Each decision option
    for c in L: # Each class label
        ind_rc = np.argwhere((decisions==r) & (labels==c))

        # Decision = Marker Shape; True Labels = Marker Color
        marker = marker_shapes[r-4]
        if r == c:
            plt.plot(X[ind_rc, 0], X[ind_rc, 1], marker+'g')
        else:
            plt.plot(X[ind_rc, 0], X[ind_rc, 1], marker+'r')

plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("2D view Classification Decisions: Marker Shape/Class, Color/Correct Labels")
plt.tight_layout()
plt.show()

################# PCA

# ========== PCA applied to data from a Uniform Mixture PDF ==========
def perform_pca(X):
    """  Principal Component Analysis (PCA) on real-valued vector data.

    Args:
        X: Real-valued matrix of samples with shape [N, n], N for sample count and n for dimensionality.

    Returns:
        U: An orthogonal matrix [n, n] that contains the PCA projection vectors, ordered from first to last.
        D: A diagonal matrix [n, n] that contains the variance of each PC corresponding to the projection vectors.
        Z: PC projection matrix of the zero-mean input samples, shape [N, n].
    """

    # First derive sample-based estimates of mean vector and covariance matrix:
    mu = np.mean(X, axis=0)
    sigma = np.cov(X.T)

    # Mean-subtraction is a necessary assumption for PCA, so perform this to obtain zero-mean sample set
    C = X - mu

    # Get the eigenvectors (in U) and eigenvalues (in D) of the estimated covariance matrix
    lambdas, U = np.linalg.eig(sigma)
    # Get the indices from sorting lambdas in order of increasing value, with ::-1 slicing to then reverse order
    idx = lambdas.argsort()[::-1]
    # Extract corresponding sorted eigenvectors and eigenvalues
    U = U[:, idx]
    D = np.diag(lambdas[idx])

    # PC projections of zero-mean samples, U^Tx (x mean-centred), matrix over N is XU
    Z = C.dot(U)

    # If using sklearn instead:
    # pca = PCA(n_components=X.shape[1])  # n_components is how many PCs we'll keep... let's take all of them
    # X_fit = pca.fit(X)  # Is a fitted estimator, not actual data to project
    # Z = pca.transform(X)

    return U, D, Z

X_UMM = X
# Perform PCA on transposed UMM variable X
_, _, Z = perform_pca(X_UMM)

# Add back mean vector to PC projections if you want PCA reconstructions
Z_UMM = Z + np.mean(X_UMM, axis=0)
print(Z_UMM.shape)
marker_shapes = '.o^1s+*'
marker_colors = 'rbgmgbrrbgmgbr' 
for r in L: # Each decision option
    for c in L: # Each class label
        ind_rc = np.argwhere((decisions==r) & (labels==c))

        # Decision = Marker Shape; True Labels = Marker Color
        marker = marker_shapes[r-4]
        if r == c:
            plt.plot(Z_UMM[ind_rc, 0], Z_UMM[ind_rc, 1], marker+'g')
        else:
            plt.plot(Z_UMM[ind_rc, 0], Z_UMM[ind_rc, 1], marker+'r')

plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("PCA of Classification Decisions: Marker Shape/Class, Color/Correct Labels")
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
marker_shapes = '.o^1s+*'
marker_colors = 'rbgmgbrrbgmgbr' 
for r in L: # Each decision option
    for c in L: # Each class label
        ind_rc = np.argwhere((decisions==r) & (labels==c))

        # Decision = Marker Shape; True Labels = Marker Color
        #marker = marker_shapes[r-4]
        if r == c:
            ax.scatter(Z_UMM[ind_rc, 0], Z_UMM[ind_rc, 1], Z_UMM[ind_rc, 2], c='g', marker = marker_shapes[r-4])
        else:
            ax.scatter(Z_UMM[ind_rc, 0], Z_UMM[ind_rc, 1], Z_UMM[ind_rc, 2], c='r', marker = marker_shapes[r-4])

plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("PCA of Classification Decisions: Marker Shape/Class, Color/Correct Labels")
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
marker_shapes = '.o^1s+*'
marker_colors = 'rbgymck' 
for r in L: # Each decision option
    for c in L: # Each class label
        ind_rc = np.argwhere((decisions==r) & (labels==c))

        # Decision = Marker Shape; True Labels = Marker Color
        #marker = marker_shapes[r-4]
        if r == c:
            ax.scatter(Z_UMM[ind_rc, 0], Z_UMM[ind_rc, 1], Z_UMM[ind_rc, 2], c=marker_colors[r-4], marker = marker_shapes[r-4], s=1)
        else:
            ax.scatter(Z_UMM[ind_rc, 0], Z_UMM[ind_rc, 1], Z_UMM[ind_rc, 2], c=marker_colors[r-4], marker = marker_shapes[r-4], s=1)

plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("PCA Dimensionality Reduction on Dataset: Colors/Classes")
plt.tight_layout()
plt.show()