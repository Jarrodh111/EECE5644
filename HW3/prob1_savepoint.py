#   Jarrod Homer
#   EECE5644 HW3

import matplotlib.pyplot as plt # For general plotting
import matplotlib.colors as mcol

import numpy as np

from math import ceil, floor

from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Utility to visualize PyTorch network and shapes
from torchsummary import summary

import time

import os


np.set_printoptions(suppress=True)

# Set seed to generate reproducible "pseudo-randomness" (handles scipy's "randomness" too)
np.random.seed(7)
torch.manual_seed(7)

plt.rc('font', size=22)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=18)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize
plt.rc('figure', titlesize=22)   # fontsize of the figure title

############################################    END Imports




############################################    Generate Data

def generate_data_from_gmm(N, pdf_params):
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
            X[indices, 0] =  norm.rvs(pdf_params['m'][l-1], pdf_params['C'][l-1], Nl)
        else:
            X[indices, :] =  mvn.rvs(pdf_params['m'][l-1], pdf_params['C'][l-1], Nl)
    
    return X, labels



# Generate dataset from two different 3D Gaussian distributions/categories
N = 10000

gmm_pdf = {}

# Class priors
gmm_pdf['priors'] = np.array([0.25, 0.25, 0.25, 0.25])
num_classes = len(gmm_pdf['priors'])
# Mean and covariance of data pdfs conditioned on labels
gmm_pdf['m'] = np.array([[1.3, 1.3, 0],
                         [-1.3, 1.3, 0],
                         [-1.3, -1.3, 0],
                         [1.3, -1.3, 0]])  # Gaussian distributions means

gmm_pdf['C'] = np.array([[[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]],
                         [[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]],
                          [[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]],
                          [[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]]])  # Gaussian distributions covariance matrices

# Plot the original data and their true labels
fig = plt.figure(figsize=(10, 10))
ax_raw = fig.add_subplot(111, projection='3d')

X_train, y_train = generate_data_from_gmm(N, gmm_pdf)
X_valid, y_valid = generate_data_from_gmm(N, gmm_pdf)

n = X_train.shape[1]
L = np.array(range(num_classes))
print(L)

# Count up the number of samples per class
N_per_l = np.array([sum(y_train == l) for l in L])
print(N_per_l)

ax_raw.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], X_train[y_train == 0, 2], c='r', label="Class 0")
ax_raw.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], X_train[y_train == 1, 2], c='b', label="Class 1")
ax_raw.scatter(X_train[y_train == 2, 0], X_train[y_train == 2, 1], X_train[y_train == 2, 2], c='g', label="Class 2")
ax_raw.scatter(X_train[y_train == 3, 0], X_train[y_train == 3, 1], X_train[y_train == 3, 2], c='m', label="Class 3")
ax_raw.set_xlabel(r"$x_1$")
ax_raw.set_ylabel(r"$x_2$")
ax_raw.set_zlabel(r"$x_3$")
# Set equal axes for 3D plots
ax_raw.set_box_aspect((np.ptp(X_train[:, 0]), np.ptp(X_train[:, 1]), np.ptp(X_train[:, 2])))

plt.title("Data and True Class Labels")
plt.legend()
plt.tight_layout()

#####################################     END Generate Data


######################################      MAP on Test Data

# Min prob. of error classifier
# Conditional likelihoods of each class given x, shape (C, N)
class_cond_likelihoods = np.array([mvn.pdf(X_valid, gmm_pdf['m'][l], gmm_pdf['C'][l]) for l in L])
# Take diag so we have (C, C) shape of priors with prior prob along diagonal
class_priors = np.diag(gmm_pdf['priors'])
# class_priors*likelihood with diagonal matrix creates a matrix of posterior probabilities
# with each class as a row and N columns for samples, e.g. row 1: [p(y1)p(x1|y1), ..., p(y1)p(xN|y1)]
class_posteriors = class_priors.dot(class_cond_likelihoods)

# MAP rule, take largest class posterior per example as your decisions matrix (N, 1)
# Careful of indexing! Added np.ones(N) just for difference in starting from 0 in Python and labels={1,2,3}
decisions = np.argmax(class_posteriors, axis=0) + 0*np.ones(N) 

# Simply using sklearn confusion matrix
print("Confusion Matrix (rows: Predicted class, columns: True class):")
conf_mat = confusion_matrix(decisions, y_valid)
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
ax_raw = fig.add_subplot(111, projection='3d')

marker_shapes = '.o^1s+*.o^1s+*'
marker_colors = 'rbgmgbrrbgmgbr' 
for r in L: # Each decision option
    for c in L: # Each class label
        ind_rc = np.argwhere((decisions==r) & (y_valid==c))

        # Decision = Marker Shape; True Labels = Marker Color
        
        if r == c:
            ax_raw.scatter(X_valid[ind_rc, 0], X_valid[ind_rc, 1], X_valid[ind_rc, 2], c='g', marker = marker_shapes[r])
        else:
            ax_raw.scatter(X_valid[ind_rc, 0], X_valid[ind_rc, 1], X_valid[ind_rc, 2], c='r', marker = marker_shapes[r])

#plt.legend()
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("2D view of Classification Decisions: Marker Shape/Class, Color/Correct Labels")
plt.tight_layout()
plt.show()




###############################################     END Map on Test Data


# Create coordinate matrices determined by the sample space
xx, yy, zz = np.meshgrid(np.linspace(-4, 4, 250), np.linspace(-4, 4, 250), np.linspace(-4, 4, 250))
grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
grid_tensor = torch.FloatTensor(grid)  

input_dim = X_train.shape[1]
output_dim = len(gmm_pdf['priors'])


n_hidden_neurons = 16



#################################################   Define NN and Algos


# Parameters like n_hidden_neurons set above
quick_two_layer_mlp = nn.Sequential(
    nn.Linear(input_dim, n_hidden_neurons),
    nn.ReLU(),
    nn.Linear(n_hidden_neurons, output_dim),
    nn.LogSoftmax(dim=1)
)
# Stochastic GD with learning rate and momentum hyperparameters
optimizer = torch.optim.SGD(quick_two_layer_mlp.parameters(), lr=0.01, momentum=0.9)
# The nn.CrossEntropyLoss() loss function automatically performs a log_softmax() to 
# the output when validating, on top of calculating the negative log-likelihood using 
# nn.NLLLoss(), while also being more stable numerically... So don't implement from scratch
criterion = nn.CrossEntropyLoss()
num_epochs = 10    #100

# Convert numpy structures to PyTorch tensors, as these are the data types required by the library
X_tensor = torch.FloatTensor(X_train)
y_tensor = torch.LongTensor(y_train)

# Forward pass through MLP
probs = quick_two_layer_mlp(X_tensor)
print("probs:")
print(probs)

# Backpropagation training insert here







X_test_tensor = torch.FloatTensor(X_valid)
y_test_tensor = torch.LongTensor(y_valid)

# Create your dataset objects
train_data = TensorDataset(X_tensor, y_tensor) 
test_data = TensorDataset(X_test_tensor, y_test_tensor) 

train_dataloader = DataLoader(train_data, batch_size=500, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64) # No need to shuffle...





def model_train_loader(model, dataloader, criterion, optimizer):
    size = len(dataloader.dataset)

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        predictions = model(X)
        loss = criterion(predictions, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Report loss every 10 batches
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
            
def model_test_loader(model, dataloader, criterion):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # Tracking test loss (cross-entropy) and correct classification rate (accuracy)
    test_loss, correct = 0, 0
    
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            predictions = model(X)
            test_loss += criterion(predictions, y)
            correct += (predictions.argmax(1) == y).type(torch.float).sum()
            
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    

# Let's train the Sequential model this time
# And look at how we're training + testing in parallel
# Useful if we wanted to do something like early stopping!
# Nesterov is a better revision of the momentum update
optimizer = torch.optim.SGD(quick_two_layer_mlp.parameters(), lr=0.01, momentum=0.9, nesterov=True)
for t in range(num_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    model_train_loader(quick_two_layer_mlp, train_dataloader, criterion, optimizer)
    model_test_loader(quick_two_layer_mlp, test_dataloader, criterion)



# Z matrix are the predictions resulting from the forward pass through the network
Z_probs = quick_two_layer_mlp(X_test_tensor).detach().numpy()
print(Z_probs.shape)
Z_pred = np.argmax(Z_probs, 1)

fig = plt.figure(figsize=(10, 10))
ax_raw = fig.add_subplot(111, projection='3d')

# uses gray background for black dots
#plt.pcolormesh(xx, yy, Z_pred, cmap=plt.cm.coolwarm)
print(Z_pred.shape)
print(Z_pred)
    
ax_raw.scatter(X_valid[y_valid==0, 0], X_valid[y_valid==0, 1], X_valid[y_valid==0, 2], 'r', label="Class 0")
ax_raw.scatter(X_valid[y_valid==1, 0], X_valid[y_valid==1, 1], X_valid[y_valid==1, 2], 'b', label="Class 1")
ax_raw.scatter(X_valid[y_valid==2, 0], X_valid[y_valid==2, 1], X_valid[y_valid==2, 2], 'g', label="Class 2")
ax_raw.scatter(X_valid[y_valid==3, 0], X_valid[y_valid==3, 1], X_valid[y_valid==3, 2], 'm', label="Class 3")
ax_raw.set_xlabel(r"$x_1$")
ax_raw.set_ylabel(r"$x_2$")
ax_raw.set_zlabel(r"$x_3$")
# Set equal axes for 3D plots
ax_raw.set_box_aspect((np.ptp(X_train[:, 0]), np.ptp(X_train[:, 1]), np.ptp(X_train[:, 2])))
plt.title("MLP Classification Boundaries Test Set")
plt.legend()
plt.show()





########################################## Saving and loading code im guessing I will use this for training different models and then evaluating



# A state_dict is simply dictionary object that maps each layer to its parameter tensor
# Only the layers with learnable parameters, as wellas the optimizer's detials, e.g. hyperparameters, are stored
# Saving the file 'model.pth' to my current working directory (cwd)
torch.save(quick_two_layer_mlp.state_dict(), os.getcwd() + '/model.pth')

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in quick_two_layer_mlp.state_dict():
    print(param_tensor, "\t", quick_two_layer_mlp.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])



# Model class has to be defined somewhere in script before loading from disk
load_model = nn.Sequential(
    nn.Linear(input_dim, n_hidden_neurons),
    nn.ReLU(),
    nn.Linear(n_hidden_neurons, output_dim),
    nn.LogSoftmax(dim=1)
)
load_model.load_state_dict(torch.load(os.getcwd() + '/model.pth'))
load_model.eval()

# Double check test set accuracy
predictions = load_model(X_test_tensor)
correct = (predictions.argmax(1) == y_test_tensor).type(torch.float).sum() / X_test_tensor.shape[0]
print("Model loaded from disk has correct classification accuracy of: {:.1f}%".format(correct.item()*100))