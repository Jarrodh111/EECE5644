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
priors = np.array([0.65, 0.35])  
# Determine number of classes/mixture components
C = len(priors)

# Mean and covariance of data pdfs conditioned on labels
mu = np.array([[-0.5, -0.5, -0.5],
               [1, 1, 1]])  # Gaussian distributions means

Sigma = np.array([[[1, -0.5, 0.3],
                   [-0.5, 1, -0.5],
                   [0.3, -0.5, 1]],

                  [[1, 0.3, -0.2],
                   [0.3, 1, 0.3],
                   [-0.2, 0.3, 1]]])  # Gaussian distributions covariance matrices

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
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
plt.plot(X[y==0, 0], X[y==0, 1], X[y==0, 2], 'bo', label="Class 0")
plt.plot(X[y==1, 0], X[y==1, 1], X[y==1, 2], 'rx', label="Class 1")

plt.legend()
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")
plt.title("Data and True Labels")
plt.tight_layout()
plt.show()

Y = np.array(range(C))  # 0-(C-1)

# We are going to use a 0-1 loss matrix for this problem
Lambda = np.ones((C, C)) - np.identity(C)
print(Lambda)

#Use Naive Approach
mu = np.array([[-0.5, -0.5, -0.5],
               [1, 1, 1]])  # Gaussian distributions means

Sigma = np.array([[[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]],

                  [[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]]])  # Gaussian distributions covariance matrices

# Calculate class-conditional likelihoods p(x|Y=j) for each label of the N observations
class_cond_likelihoods = np.array([multivariate_normal.pdf(X, mu[j], Sigma[j]) for j in Y])
class_priors = np.diag(priors)
print(class_cond_likelihoods)
print(class_cond_likelihoods.shape)
print(class_priors.shape)
class_posteriors = class_priors.dot(class_cond_likelihoods)
print(class_posteriors)


##################    Generate Different Gamma    ##################
class_cond_ratio=np.zeros(N)
for i in range(N):
    class_cond_ratio[i]=class_cond_likelihoods[0][i]/class_cond_likelihoods[1][i]

gamma_decisions=np.zeros(N)

print("\nTesting Gammas")
fig = plt.figure(figsize=(12, 10))

min_test_prob_error=1
best_gamma=-1
test_gamma= np.append((np.arange(0,4,.01)),(np.arange(4,100,1)))
test_gamma= np.append(test_gamma,[10,100,1000,10000,100000,1000000,1000000000,100000000000000])
true_pos_prob=np.zeros(len(test_gamma))
false_pos_prob=np.zeros(len(test_gamma))
false_neg_prob=np.zeros(len(test_gamma))
true_neg_prob=np.zeros(len(test_gamma))
test_prob_error=np.zeros(len(test_gamma))
gindex=0
for g in test_gamma:
    gamma_conf_mat = np.zeros((C, C))
    for j in range(N):
        if(class_cond_likelihoods[1][j]>class_cond_likelihoods[0][j]*g):
            gamma_decisions[j]=1
        else:
            gamma_decisions[j]=0

        #TP
        if(gamma_decisions[j]==1 and y[j]==1):
            gamma_conf_mat[0][0]+=1
        #FP
        if(gamma_decisions[j]==1 and y[j]==0):
            gamma_conf_mat[0][1]+=1
        #FN
        if(gamma_decisions[j]==0 and y[j]==1):
            gamma_conf_mat[1][0]+=1
        #TN
        if(gamma_decisions[j]==0 and y[j]==0):
            gamma_conf_mat[1][1]+=1
    true_pos_prob[gindex]=gamma_conf_mat[0][0]/(gamma_conf_mat[0][0]+gamma_conf_mat[1][0])
    false_pos_prob[gindex]=gamma_conf_mat[0][1]/(gamma_conf_mat[0][1]+gamma_conf_mat[1][1])
    false_neg_prob[gindex]=1-true_pos_prob[gindex]
    true_neg_prob[gindex]=1-false_neg_prob[gindex]
    test_prob_error[gindex]=false_pos_prob[gindex]*priors[0]+false_neg_prob[gindex]*priors[1]
    if(test_prob_error[gindex]<min_test_prob_error):
        min_test_prob_error=test_prob_error[gindex]
        best_gamma=g
    gindex+=1
plt.plot(false_pos_prob, true_pos_prob, 'bo')
bestgind=np.argmin(test_prob_error)
plt.plot(false_pos_prob[bestgind], true_pos_prob[bestgind], 'go', markersize=15, label="Optimal Gamma = " +str(best_gamma))
plt.text(0.1, 0.8, s="("+str(round(false_pos_prob[bestgind],3)) + ", "+str(round(true_pos_prob[bestgind],3))+")")
plt.text(0.5, 0.5, s="min prob. of error = " + str(round(min_test_prob_error,3)))
print(best_gamma,min_test_prob_error)
plt.legend()
plt.xlabel("False Positive Probability")
plt.ylabel("True Positive Probability")
plt.ylim(-0.1, 1.1)
plt.xlim(-0.1, 1.1)
plt.title("ROC Curve Part B")
plt.tight_layout()
plt.show()
##################    End Generate Different Gamma    ##################

# We want to create the risk matrix of size 3 x N 
cond_risk = Lambda.dot(class_posteriors)
print(cond_risk)

# Get the decision for each column in risk_mat
decisions = np.argmin(cond_risk, axis=0)
print(decisions.shape)

# Plot for decisions vs true labels
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(1, 1, 1, projection='3d')

marker_shapes = 'ox+*.' # Accomodates up to C=5
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
        if i == j:
            ax.scatter(X[ind_ij, 0], X[ind_ij, 1], X[ind_ij, 2], marker_shapes[j]+'g', label="Correct Class "+str(j))

        if i != j:
            ax.scatter(X[ind_ij, 0], X[ind_ij, 1], X[ind_ij, 2], marker, s=500, label="Incorrect Class "+str(j))
            
print("Confusion matrix:")
print(conf_mat)

print("Minimum Probability of Error:")
prob_error = 1 - np.diag(conf_mat).dot(sample_class_counts / N)
print(prob_error)

plt.legend()
ax.set_xlabel("z-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")
plt.title("Minimum Probability of Error Classified Sampled Data Part B:  {:.3f}".format(prob_error))
plt.show()