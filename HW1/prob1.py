# EECE5644 HW 1
# Jarrod Homer
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 20, 100)  # Create a list of evenly-spaced numbers over the range
plt.plot(x, np.sin(x))       # Plot the sine of each x point
plt.show()                   # Display the plot

N = 10000

# Mean and covariance of data pdfs conditioned on labels
mu = np.array([[-1, 0],
               [1, 0],
               [0, 1]])  # Gaussian distributions means
Sigma = np.array([[[1, -0.4],
                   [-0.4, 0.5]],
                  [[0.5, 0],
                   [0, 0.2]],
                  [[0.1, 0],
                   [0, 0.1]]])  # Gaussian distributions covariance matrices
# Determine dimensionality from mixture PDF parameters
n = mu.shape[1]

# Class priors
priors = np.array([0.15, 0.35, 0.5])  
C = len(priors)
# Decide randomly which samples will come from each component
u = np.random.rand(N)
thresholds = np.cumsum(priors)
thresholds = np.insert(thresholds, 0, 0) # For intervals of classes

# Output samples and labels
X = np.zeros([N, n])
labels = np.zeros(N) # KEEP TRACK OF THIS

# Plot for original data and their true labels
fig = plt.figure(figsize=(12, 10))
marker_shapes = 'd+.'
marker_colors = 'rbg' 

L = np.array(range(1, C+1))
for l in L:
    # Get randomly sampled indices for this component
    indices = np.argwhere((thresholds[l-1] <= u) & (u <= thresholds[l]))[:, 0]
    # No. of samples in this component
    Nl = len(indices)  
    labels[indices] = l * np.ones(Nl)
    X[indices, :] =  multivariate_normal.rvs(mu[l-1], Sigma[l-1], Nl)
    plt.plot(X[labels==l, 0], X[labels==l, 1], marker_shapes[l-1] + marker_colors[l-1], label="True Class {}".format(l))

    
# Plot the original data and their true labels
plt.legend()
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("Generated Original Data Samples")
plt.tight_layout()
plt.show()