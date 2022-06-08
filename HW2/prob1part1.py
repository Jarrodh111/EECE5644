# Widget to manipulate plots in Jupyter notebooks


from sys import float_info  # Threshold smallest positive floating value

import matplotlib.pyplot as plt # For general plotting

import numpy as np
import pandas as pd

from scipy.stats import multivariate_normal
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

np.set_printoptions(suppress=True)

# Set seed to generate reproducible "pseudo-randomness" (handles scipy's "randomness" too)
np.random.seed(7)

plt.rc('font', size=22)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=18)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize
plt.rc('figure', titlesize=22)   # fontsize of the figure title

def generate_data_from_gmm(N, pdf_params, fig_ax=None):
    # Determine dimensionality from mixture PDF parameters
    n = pdf_params['m'].shape[1]
    # Output samples and labels
    X = np.zeros([N, n])
    labels = np.zeros(N)
    
    # Decide randomly which samples will come from each component
    u = np.random.rand(N)
    thresholds = np.cumsum(pdf_params['priors'])
    thresholds = np.insert(thresholds, 0, 0) # For intervals of classes

    L = np.array(range(1, len(pdf_params['priors'])+1))
    for l in L:
        # Get randomly sampled indices for this component
        indices = np.argwhere((thresholds[l-1] <= u) & (u <= thresholds[l]))[:, 0]
        # No. of samples in this component
        Nl = len(indices)  
        labels[indices] = l * np.ones(Nl) - 1
        if n == 1:
            X[indices, 0] =  norm.rvs(pdf_params['m'][l-1], pdf_params['C'][l-1], Ny)
        else:
            X[indices, :] =  multivariate_normal.rvs(pdf_params['m'][l-1], pdf_params['C'][l-1], Nl)
    
    return X, labels

# Generate ROC curve samples
def estimate_roc(discriminant_score, labels):
    N_labels = np.array((sum(labels == 0), sum(labels == 1)))

    # Sorting necessary so the resulting FPR and TPR axes plot threshold probabilities in order as a line
    sorted_score = sorted(discriminant_score)

    # Use gamma values that will account for every possible classification split
    gammas = ([sorted_score[0] - float_info.epsilon] +
              sorted_score +
              [sorted_score[-1] + float_info.epsilon])

    # Calculate the decision label for each observation for each gamma
    decisions = [discriminant_score >= g for g in gammas]
    
    # Retrieve indices where FPs occur
    ind10 = [np.argwhere((d == 1) & (labels == 0)) for d in decisions]
    # Compute FP rates (FPR) as a fraction of total samples in the negative class
    p10 = [len(inds) / N_labels[0] for inds in ind10]
    # Retrieve indices where TPs occur
    ind11 = [np.argwhere((d == 1) & (labels == 1)) for d in decisions]
    # Compute TP rates (TPR) as a fraction of total samples in the positive class
    p11 = [len(inds) / N_labels[1] for inds in ind11]
    
    # ROC has FPR on the x-axis and TPR on the y-axis, but return others as well for convenience
    roc = {}
    roc['p10'] = np.array(p10)
    roc['p11'] = np.array(p11)

    return roc, gammas

def get_binary_classification_metrics(predictions, labels):
    N_labels = np.array((sum(labels == 0), sum(labels == 1)))

    # Get indices and probability estimates of the four decision scenarios:
    # (true negative, false positive, false negative, true positive)
    class_metrics = {}
    
    # True Negative Probability Rate
    ind_00 = np.argwhere((predictions == 0) & (labels == 0))
    class_metrics['tnr'] = len(ind_00) / N_labels[0]
    # False Positive Probability Rate
    ind_10 = np.argwhere((predictions == 1) & (labels == 0))
    class_metrics['fpr'] = len(ind_10) / N_labels[0]
    # False Negative Probability Rate
    ind_01 = np.argwhere((predictions == 0) & (labels == 1))
    class_metrics['fnr'] = len(ind_01) / N_labels[1]
    # True Positive Probability Rate
    ind_11 = np.argwhere((predictions == 1) & (labels == 1))
    class_metrics['tpr'] = len(ind_11) / N_labels[1]

    return class_metrics

# Generate dataset from two different 3D Gaussian distributions/categories
N = 10000

gmm_pdf = {}

# Class priors
gmm_pdf['priors'] = np.array([0.325, 0.325, 0.35])
# Mean and covariance of data pdfs conditioned on labels
gmm_pdf['m'] = np.array([[3, 0],
                         [0, 3],
                         [2, 2]])  # Gaussian distributions means
gmm_pdf['C'] = np.array([[[2, 0],
                          [0, 1]],
                         [[1, 0],
                          [0, 2]],
                         [[1, 0],
                          [0, 1]]])  # Gaussian distributions covariance matrices

# Plot the original data and their true labels
fig = plt.figure(figsize=(10, 10))




X, labels = generate_data_from_gmm(N, gmm_pdf)
oldlabels=labels
print(labels)
for l in range(N):
    if labels[l]==1:
        labels[l]=0
    if labels[l]==2:
        labels[l]=1

print(sum(labels))
print(labels)



C = len(gmm_pdf['priors'])-1
Y = np.array(range(C))  # 0-(C-1)
num_classes = len(gmm_pdf['priors'])


n = X.shape[1]
L = np.array(range(num_classes))

# Count up the number of samples per class
N_per_l = np.array([sum(labels == 0),sum(labels == 1)])
print(N_per_l)

plt.plot(X[labels == 0, 0], X[labels == 0, 1], 'om', label="Class 0")
plt.plot(X[labels == 1, 0], X[labels == 1, 1], 'ob', label="Class 1")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
# Set equal axes for 3D plots

plt.title("Data and True Class Labels")
plt.legend()
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# Compute class conditional likelihoods to express ratio test, where ratio is discriminant score
class_conditional_likelihoods = np.array([multivariate_normal.pdf(X, gmm_pdf['m'][l], gmm_pdf['C'][l]) for l in L])
class_conditional_likelihoods[0]=class_conditional_likelihoods[0]+class_conditional_likelihoods[1]
class_conditional_likelihoods[1]=class_conditional_likelihoods[2]
class_conditional_likelihoods=class_conditional_likelihoods[:2]
print("owa")
print(class_conditional_likelihoods.shape)
# Class conditional log likelihoods equate to decision boundary log gamma in the 0-1 loss case
discriminant_score_erm = np.log(class_conditional_likelihoods[1]) - np.log(class_conditional_likelihoods[0])

# Construct the ROC for ERM by changing log(gamma)
roc_erm, gammas_empirical = estimate_roc(discriminant_score_erm, labels)
# roc_erm returns a np.array of shape(2, N+2) where N+2 are the number of thresholds
# and 2 rows are the FPR and TPR respectively

#plt.ioff() # Interactive plotting off
fig_roc, ax_roc = plt.subplots(figsize=(10, 10));
#plt.ion()

ax_roc.plot(roc_erm['p10'], roc_erm['p11'], label="Empirical ERM Classifier ROC Curve")
ax_roc.set_xlabel(r"Probability of False Alarm $P(D=1|L=0)$")
ax_roc.set_ylabel(r"Probability of True Positive $P(D=1|L=1)$")

plt.grid(True)
#display(fig_roc)
fig_roc;




# ROC returns FPR vs TPR, but prob error needs FNR so take 1-TPR
# P(error; γ) = P(D = 1|L = 0; γ)P(L = 0)+P(D = 0|L = 1; γ)P(L = 1)
prob_error_empirical = np.array((roc_erm['p10'], 1 - roc_erm['p11'])).T.dot(N_per_l / N)

# Min prob error for the empirically-selected gamma thresholds
min_prob_error_empirical = np.min(prob_error_empirical)
min_ind_empirical = np.argmin(prob_error_empirical)

# Compute theoretical gamma as log-ratio of priors (0-1 loss) -> MAP classification rule
gamma_map = 0.65 / 0.35
decisions_map = discriminant_score_erm >= np.log(gamma_map)

class_metrics_map = get_binary_classification_metrics(decisions_map, labels)
# To compute probability of error, we need FPR and FNR
min_prob_error_map = np.array((class_metrics_map['fpr'] * 0.65 + 
                               class_metrics_map['fnr'] * 0.35))

# Plot theoretical and empirical
ax_roc.plot(roc_erm['p10'][min_ind_empirical], roc_erm['p11'][min_ind_empirical], 'go', label="Empirical Min P(Error) ERM",
            markersize=14)
ax_roc.plot(class_metrics_map['fpr'], class_metrics_map['tpr'], 'rx', label="Theoretical Min P(Error) ERM", markersize=14)
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')

print("Min Empirical P(error) for ERM = {:.3f}".format(min_prob_error_empirical))
print("Min Empirical Gamma = {:.3f}".format(np.exp(gammas_empirical[min_ind_empirical])))

print("Min Theoretical P(error) for ERM = {:.3f}".format(min_prob_error_map))
print("Min Theoretical Gamma = {:.3f}".format(gamma_map))

plt.title("ROC Curve Part 1")
plt.show()


# Simply using sklearn confusion matrix
print("Confusion Matrix (rows: Predicted class, columns: True class):")
conf_mat = confusion_matrix(decisions_map, labels)
print(conf_mat)
correct_class_samples = np.sum(np.diag(conf_mat))
print("Total Mumber of Misclassified Samples: {:d}".format(N - correct_class_samples))
prob_error = 1 - (correct_class_samples / N)
print("Empirically Estimated Probability of Error: {:.4f}".format(prob_error))


fig_disc_grid, ax_disc = plt.subplots(figsize=(10, 10));
marker_shapes = '.^s*' # Accomodates up to C=5
marker_colors = 'mbyr'
for i in Y: # Each decision option
    for j in Y: # Each class label
        ind_ij = np.argwhere((decisions_map==i) & (labels==j))
        #conf_mat[i, j] = len(ind_ij)/sample_class_counts[j] # Average over class sample count

        # True label = Marker shape; Decision = Marker Color
        marker = marker_shapes[j] + marker_colors[i]
        if i == j:
            plt.plot(X[ind_ij, 0], X[ind_ij, 1], marker, markersize=5, markerfacecolor='none', label="Correct Class "+str(i))

        marker = marker_shapes[j+2] + marker_colors[i+2]
        if i != j:
            plt.plot(X[ind_ij, 0], X[ind_ij, 1], marker, markersize=5, markerfacecolor='none', label="Incorrect Class "+str(i))


################################### add countour


horizontal_grid = np.linspace(np.floor(np.min(X[:,0])), np.ceil(np.max(X[:,0])), 100)
vertical_grid = np.linspace(np.floor(np.min(X[:,1])), np.ceil(np.max(X[:,1])), 100)

# Generate a grid of scores that spans the full range of data 
[h, v] = np.meshgrid(horizontal_grid, vertical_grid)
# Flattening to feed vectorized matrix in pdf evaluation
gridxy = np.array([h.reshape(-1), v.reshape(-1)])
likelihood_grid_vals = np.array([multivariate_normal.pdf(gridxy.T, gmm_pdf['m'][0], gmm_pdf['C'][0]) + multivariate_normal.pdf(gridxy.T,
 gmm_pdf['m'][1], gmm_pdf['C'][1]), multivariate_normal.pdf(gridxy.T, gmm_pdf['m'][2], gmm_pdf['C'][2])])
# Where a score of 0 indicates decision boundary level
print(likelihood_grid_vals.shape)
discriminant_score_grid_vals = np.log(likelihood_grid_vals[1]) - np.log(likelihood_grid_vals[0]) - np.log(gamma_map)

# Contour plot of decision boundaries
discriminant_score_grid_vals = np.array(discriminant_score_grid_vals).reshape(100, 100)
equal_levels = np.array((0.3, 0.6, 0.9))
min_DSGV = np.min(discriminant_score_grid_vals) * equal_levels[::-1]
max_DSGV = np.max(discriminant_score_grid_vals) * equal_levels
contour_levels = min_DSGV.tolist() + [0] + max_DSGV.tolist()
cs = ax_disc.contour(horizontal_grid, vertical_grid, discriminant_score_grid_vals.tolist(), contour_levels, colors='k')
ax_disc.clabel(cs, fontsize=16, inline=1)

plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
# Set equal axes for 3D plots

plt.title("MAP Decisions")
plt.legend()
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

