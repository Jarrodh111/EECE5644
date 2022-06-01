# EECE5644 HW 1
# Jarrod Homer

import matplotlib.pyplot as plt # For general plotting
from matplotlib import cm

import numpy as np

from sys import float_info # Threshold smallest positive floating value

from scipy.stats import multivariate_normal # MVN not univariate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

# Decide randomly which samples will come from each component (taking class 1 from standard normal values above 0.35)
labels = np.random.rand(N) >= priors[0]
L = np.array(range(C))
Nl = np.array([sum(labels == l) for l in L])

# Draw samples from each class pdf
X = np.zeros((N, n))
X[labels == 0, :] =  multivariate_normal.rvs(mu[0], Sigma[0], Nl[0])
X[labels == 1, :] =  multivariate_normal.rvs(mu[1], Sigma[1], Nl[1])


# Plot the original data and their true labels
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
plt.plot(X[labels==0, 0], X[labels==0, 1], X[labels==0, 2], 'bo', label="Class 0")
plt.plot(X[labels==1, 0], X[labels==1, 1], X[labels==1, 2], 'rx', label="Class 1")

plt.legend()
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")
plt.title("Data and True Class Labels")
plt.tight_layout()
plt.show()

# MAP classifier (is a special case of ERM corresponding to 0-1 loss)
# 0-1 loss values yield MAP decision rule
Lambda = np.ones((C, C)) - np.identity(C)
print(Lambda)

# Expected Risk Minimization Classifier (using true model parameters)
# In practice the parameters would be estimated from training samples
# Using log-likelihood-ratio as the discriminant score for ERM
class_conditional_likelihoods = np.array([multivariate_normal.pdf(X, mu[l], Sigma[l]) for l in L])
discriminant_score_erm = np.log(class_conditional_likelihoods[1]) - np.log(class_conditional_likelihoods[0])

# Gamma threshold for MAP decision rule (remove Lambdas and you obtain same gamma on priors only; 0-1 loss simplification)
gamma_map = (Lambda[1,0] - Lambda[0,0]) / (Lambda[0,1] - Lambda[1,1]) * priors[0]/priors[1]
# Same as:
# gamma_map = priors[0]/priors[1]
print(gamma_map)

decisions_map = discriminant_score_erm >= np.log(gamma_map)

# Get indices and probability estimates of the four decision scenarios:
# (true negative, false positive, false negative, true positive)

# True Negative Probability
ind_00_map = np.argwhere((decisions_map==0) & (labels==0))
p_00_map = len(ind_00_map) / Nl[0]
# False Positive Probability
ind_10_map = np.argwhere((decisions_map==1) & (labels==0))
p_10_map = len(ind_10_map) / Nl[0]
# False Negative Probability
ind_01_map = np.argwhere((decisions_map==0) & (labels==1))
p_01_map = len(ind_01_map) / Nl[1]
# True Positive Probability
ind_11_map = np.argwhere((decisions_map==1) & (labels==1))
p_11_map = len(ind_11_map) / Nl[1]

# Simply using sklearn confusion matrix
print("Confusion Matrix (rows: Predicted class, columns: True class):")
conf_mat = confusion_matrix(decisions_map, labels)
print(conf_mat)
correct_class_samples = np.sum(np.diag(conf_mat))
print("Total Mumber of Misclassified Samples: {:d}".format(N - correct_class_samples))
prob_error = 1 - (correct_class_samples / N)
print("Empirically Estimated Probability of Error: {:.4f}".format(prob_error))

# Probability of error for MAP classifier, empirically estimated
prob_error_erm = np.array((p_10_map, p_01_map)).dot(Nl.T / N)
print(np.array((p_10_map, p_01_map)).shape)
# Display MAP decisions
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# class 0 circle, class 1 +, correct green, incorrect red
ax.scatter(X[ind_00_map, 0], X[ind_00_map, 1], X[ind_00_map, 2], 'b', label="Correct Class 0")
ax.scatter(X[ind_10_map, 0], X[ind_10_map, 1], X[ind_10_map, 2], 'g', s=500, label="Incorrect Class 0")
ax.scatter(X[ind_01_map, 0], X[ind_01_map, 1], X[ind_01_map, 2], 'o', s=500, label="Incorrect Class 1")
ax.scatter(X[ind_11_map, 0], X[ind_11_map, 1], X[ind_11_map, 2], 'r', label="Correct Class 1")

plt.legend()
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")
plt.title("Part A ERM Decisions")
plt.tight_layout()
plt.show()

# Generate ROC curve samples
def estimate_roc(discriminant_score, label):
    Nlabels = np.array((sum(label == 0), sum(label == 1)))

    sorted_score = sorted(discriminant_score)

    # Use tau values that will account for every possible classification split
    taus = ([sorted_score[0] - float_info.epsilon] + 
             sorted_score +
             [sorted_score[-1] + float_info.epsilon])

    # Calculate the decision label for each observation for each gamma
    decisions = [discriminant_score >= t for t in taus]

    ind10 = [np.argwhere((d==1) & (label==0)) for d in decisions]
    p10 = [len(inds)/Nlabels[0] for inds in ind10]
    ind11 = [np.argwhere((d==1) & (label==1)) for d in decisions]
    p11 = [len(inds)/Nlabels[1] for inds in ind11]

    # ROC has FPR on the x-axis and TPR on the y-axis
    roc = np.array((p10, p11))

    return roc, taus

# Construct the ROC for ERM by changing log(gamma)
roc_erm, _ = estimate_roc(discriminant_score_erm, labels)
roc_map = np.array((p_10_map, p_11_map))

fig = plt.figure(figsize=(12, 10))
plt.plot(roc_erm[0], roc_erm[1])
plt.plot(roc_map[0], roc_map[1], 'rx', label="Minimum P(Error) MAP", markersize=16)
plt.text(0.5, 0.4, s="min prob. of error = " + str(round(prob_error_erm,3)))
plt.text(0.1, 0.8, s="("+str(round(roc_map[0],3)) + ", "+str(round(roc_map[1],3))+")")
plt.legend()
plt.xlabel(r"Probability of false alarm $P(D=1|L=0)$")
plt.ylabel(r"Probability of correct decision $P(D=1|L=1)$")
plt.title("ROC Curve Part A")
plt.grid(True)

plt.show()

##################    Generate Different Gamma    ##################
class_cond_ratio=np.zeros(N)
for i in range(N):
    class_cond_ratio[i]=class_conditional_likelihoods[0][i]/class_conditional_likelihoods[1][i]

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
        if(class_conditional_likelihoods[1][j]>class_conditional_likelihoods[0][j]*g):
            gamma_decisions[j]=1
        else:
            gamma_decisions[j]=0

        #TP
        if(gamma_decisions[j]==1 and labels[j]==1):
            gamma_conf_mat[0][0]+=1
        #FP
        if(gamma_decisions[j]==1 and labels[j]==0):
            gamma_conf_mat[0][1]+=1
        #FN
        if(gamma_decisions[j]==0 and labels[j]==1):
            gamma_conf_mat[1][0]+=1
        #TN
        if(gamma_decisions[j]==0 and labels[j]==0):
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
plt.title("ROC Curve Part A")
plt.tight_layout()
plt.show()
##################    End Generate Different Gamma    ##################
##################   Part B

# Expected Risk Minimization Classifier (using true model parameters)
# In practice the parameters would be estimated from training samples
# Using log-likelihood-ratio as the discriminant score for ERM
Sigma = np.array([[[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]],

                  [[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]]])  # Gaussian distributions covariance matrices

class_conditional_likelihoods = np.array([multivariate_normal.pdf(X, mu[l], Sigma[l]) for l in L])
discriminant_score_erm = np.log(class_conditional_likelihoods[1]) - np.log(class_conditional_likelihoods[0])

# Gamma threshold for MAP decision rule (remove Lambdas and you obtain same gamma on priors only; 0-1 loss simplification)
gamma_map = (Lambda[1,0] - Lambda[0,0]) / (Lambda[0,1] - Lambda[1,1]) * priors[0]/priors[1]
# Same as:
# gamma_map = priors[0]/priors[1]
print(gamma_map)

decisions_map = discriminant_score_erm >= np.log(gamma_map)

# Get indices and probability estimates of the four decision scenarios:
# (true negative, false positive, false negative, true positive)

# True Negative Probability
ind_00_map = np.argwhere((decisions_map==0) & (labels==0))
p_00_map = len(ind_00_map) / Nl[0]
# False Positive Probability
ind_10_map = np.argwhere((decisions_map==1) & (labels==0))
p_10_map = len(ind_10_map) / Nl[0]
# False Negative Probability
ind_01_map = np.argwhere((decisions_map==0) & (labels==1))
p_01_map = len(ind_01_map) / Nl[1]
# True Positive Probability
ind_11_map = np.argwhere((decisions_map==1) & (labels==1))
p_11_map = len(ind_11_map) / Nl[1]

# Simply using sklearn confusion matrix
print("Confusion Matrix (rows: Predicted class, columns: True class):")
conf_mat = confusion_matrix(decisions_map, labels)
print(conf_mat)
correct_class_samples = np.sum(np.diag(conf_mat))
print("Total Mumber of Misclassified Samples: {:d}".format(N - correct_class_samples))
prob_error = 1 - (correct_class_samples / N)
print("Empirically Estimated Probability of Error: {:.4f}".format(prob_error))

# Probability of error for MAP classifier, empirically estimated
prob_error_erm = np.array((p_10_map, p_01_map)).dot(Nl.T / N)
print(np.array((p_10_map, p_01_map)).shape)
# Display MAP decisions
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# class 0 circle, class 1 +, correct green, incorrect red
ax.scatter(X[ind_00_map, 0], X[ind_00_map, 1], X[ind_00_map, 2], 'b', label="Correct Class 0")
ax.scatter(X[ind_10_map, 0], X[ind_10_map, 1], X[ind_10_map, 2], 'g', s=500, label="Incorrect Class 0")
ax.scatter(X[ind_01_map, 0], X[ind_01_map, 1], X[ind_01_map, 2], 'o', s=500, label="Incorrect Class 1")
ax.scatter(X[ind_11_map, 0], X[ind_11_map, 1], X[ind_11_map, 2], 'r', label="Correct Class 1")

plt.legend()
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")
plt.title("Part B ERM Decisions")
plt.tight_layout()
plt.show()

# Construct the ROC for ERM by changing log(gamma)
roc_erm, _ = estimate_roc(discriminant_score_erm, labels)
roc_map = np.array((p_10_map, p_11_map))

fig = plt.figure(figsize=(12, 10))
plt.plot(roc_erm[0], roc_erm[1])
plt.plot(roc_map[0], roc_map[1], 'rx', label="Minimum P(Error) MAP", markersize=16)
plt.text(0.5, 0.4, s="min prob. of error = " + str(round(prob_error_erm,3)))
plt.text(0.1, 0.8, s="("+str(round(roc_map[0],3)) + ", "+str(round(roc_map[1],3))+")")
plt.legend()
plt.xlabel(r"Probability of false alarm $P(D=1|L=0)$")
plt.ylabel(r"Probability of correct decision $P(D=1|L=1)$")
plt.title("ROC Curve Part B")
plt.grid(True)

plt.show()
##################    Generate Different Gamma    ##################
class_cond_ratio=np.zeros(N)
for i in range(N):
    class_cond_ratio[i]=class_conditional_likelihoods[0][i]/class_conditional_likelihoods[1][i]

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
        if(class_conditional_likelihoods[1][j]>class_conditional_likelihoods[0][j]*g):
            gamma_decisions[j]=1
        else:
            gamma_decisions[j]=0

        #TP
        if(gamma_decisions[j]==1 and labels[j]==1):
            gamma_conf_mat[0][0]+=1
        #FP
        if(gamma_decisions[j]==1 and labels[j]==0):
            gamma_conf_mat[0][1]+=1
        #FN
        if(gamma_decisions[j]==0 and labels[j]==1):
            gamma_conf_mat[1][0]+=1
        #TN
        if(gamma_decisions[j]==0 and labels[j]==0):
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
##############  PART C

lda = LinearDiscriminantAnalysis()
X_fit = lda.fit(X, labels)  # Is a fitted estimator, not actual data to project
discriminant_score_lda = lda.transform(X).flatten()
w = X_fit.coef_[0]

# Estimate the ROC curve for this LDA classifier
roc_lda, tau_lda = estimate_roc(discriminant_score_lda, labels)

# ROC returns FPR vs TPR, but prob error needs FNR so take 1-TPR
prob_error_lda = np.array((roc_lda[0,:], 1 - roc_lda[1,:])).T.dot(Nl.T / N)

# Min prob error
min_prob_error_lda = np.min(prob_error_lda)
min_ind = np.argmin(prob_error_lda)

# Display the estimated ROC curve for LDA and indicate the operating points
# with smallest empirical error probability estimates (could be multiple)
fig = plt.figure(figsize=(12, 10))
plt.plot(roc_lda[0], roc_lda[1], 'b:')
plt.plot(roc_lda[0, min_ind], roc_lda[1, min_ind], 'r.', label="Minimum P(Error) LDA", markersize=16)
plt.text(0.5, 0.4, s="min prob. of error = " + str(round(min_prob_error_lda,3)))
plt.text(0.1, 0.8, s="("+str(round(roc_map[0],3)) + ", "+str(round(roc_map[1],3))+")")
plt.title("ROC Curve Part C")
plt.legend()

plt.show()


# Use min-error threshold
decisions_lda = discriminant_score_lda >= tau_lda[min_ind]

# Get indices and probability estimates of the four decision scenarios:
# (true negative, false positive, false negative, true positive)

# True Negative Probability
ind_00_lda = np.argwhere((decisions_lda==0) & (labels==0))
p_00_lda = len(ind_00_lda) / Nl[0]
# False Positive Probability
ind_10_lda = np.argwhere((decisions_lda==1) & (labels==0))
p_10_lda = len(ind_10_lda) / Nl[0]
# False Negative Probability
ind_01_lda = np.argwhere((decisions_lda==0) & (labels==1))
p_01_lda = len(ind_01_lda) / Nl[1]
# True Positive Probability
ind_11_lda = np.argwhere((decisions_lda==1) & (labels==1))
p_11_lda = len(ind_11_lda) / Nl[1]

# Simply using sklearn confusion matrix
print("Confusion Matrix (rows: Predicted class, columns: True class):")
conf_mat = confusion_matrix(decisions_lda, labels)
print(conf_mat)
correct_class_samples = np.sum(np.diag(conf_mat))
print("Total Mumber of Misclassified Samples: {:d}".format(N - correct_class_samples))
prob_error = 1 - (correct_class_samples / N)
print("Empirically Estimated Probability of Error: {:.4f}".format(prob_error))

# Display LDA decisions
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# class 0 circle, class 1 +, correct green, incorrect red
ax.scatter(X[ind_00_lda, 0], X[ind_00_lda, 1], X[ind_00_lda, 2], 'r', label="Correct Class 0")
ax.scatter(X[ind_10_lda, 0], X[ind_10_lda, 1], X[ind_10_lda, 2], 'o', s=500, label="Incorrect Class 0")
ax.scatter(X[ind_01_lda, 0], X[ind_01_lda, 1], X[ind_01_lda, 2], 'g', s=500, label="Incorrect Class 1")
ax.scatter(X[ind_11_lda, 0], X[ind_11_lda, 1], X[ind_11_lda, 2], 'b', label="Correct Class 1")

plt.legend()
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("LDA Part C Decisions")
plt.tight_layout()
plt.show()

print("Smallest P(error) for ERM = {}".format(prob_error_erm))
print("Smallest P(error) for LDA = {}".format(min_prob_error_lda))

##################    Generate Different Gamma    ##################
class_cond_ratio=np.zeros(N)
for i in range(N):
    class_cond_ratio[i]=class_conditional_likelihoods[0][i]/class_conditional_likelihoods[1][i]

gamma_decisions=np.zeros(N)

print("\nTesting Gammas")
fig = plt.figure(figsize=(12, 10))

min_test_prob_error=1
best_gamma=-1
test_gamma= np.array(np.arange(-5,5,.01))
test_gamma= np.append(test_gamma,[10,100,1000,10000,100000,1000000,1000000000,100000000000000])
test_gamma= np.append(test_gamma,[-10,-100,-1000,-10000,-100000,-1000000,-1000000000,-100000000000000])
true_pos_prob=np.zeros(len(test_gamma))
false_pos_prob=np.zeros(len(test_gamma))
false_neg_prob=np.zeros(len(test_gamma))
true_neg_prob=np.zeros(len(test_gamma))
test_prob_error=np.zeros(len(test_gamma))
gindex=0
for g in test_gamma:
    gamma_conf_mat = np.zeros((C, C))
    for j in range(N):
        if(discriminant_score_lda[j] >= g):
            gamma_decisions[j]=1
        else:
            gamma_decisions[j]=0

        #TP
        if(gamma_decisions[j]==1 and labels[j]==1):
            gamma_conf_mat[0][0]+=1
        #FP
        if(gamma_decisions[j]==1 and labels[j]==0):
            gamma_conf_mat[0][1]+=1
        #FN
        if(gamma_decisions[j]==0 and labels[j]==1):
            gamma_conf_mat[1][0]+=1
        #TN
        if(gamma_decisions[j]==0 and labels[j]==0):
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
plt.plot(false_pos_prob[bestgind], true_pos_prob[bestgind], 'go', markersize=15, label="Optimal Tau = " +str(round(best_gamma, 3)))
plt.text(0.1, 0.8, s="("+str(round(false_pos_prob[bestgind],3)) + ", "+str(round(true_pos_prob[bestgind],3))+")")
plt.text(0.5, 0.5, s="min prob. of error = " + str(round(min_test_prob_error,3)))
print(best_gamma,min_test_prob_error)
plt.legend()
plt.xlabel("False Positive Probability")
plt.ylabel("True Positive Probability")
plt.ylim(-0.1, 1.1)
plt.xlim(-0.1, 1.1)
plt.title("ROC Curve Part C")
plt.tight_layout()
plt.show()
##################    End Generate Different Gamma    ##################