#   Jarrod Homer
#   EECE5644 HW3

import matplotlib.pyplot as plt # For general plotting
import matplotlib.colors as mcol

import numpy as np

from scipy.stats import multivariate_normal as mvn
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.nn.functional as F

# Utility to visualize PyTorch network and shapes
from torchsummary import summary

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

N = 100

def create_data(N, noise=0.1):
    # Uses the same covariance matrix, scaled identity, for all Gaussians
    Sigma = noise * np.eye(2)
    # Five gaussian means specified to span a square and its centre
    Gs = [
        mvn(mean=[2, 2], cov=Sigma),
        mvn(mean=[-2, -2], cov=Sigma),
        mvn(mean=[2, -2], cov=Sigma),
        mvn(mean=[-2, 2], cov=Sigma),
        mvn(mean=[0, 0], cov=Sigma),
    ]
    # Draw random variable samples and assign labels, note class 3 has less samples altogether
    X = np.concatenate([G.rvs(size=N) for G in Gs])
    y = np.concatenate((np.zeros(N), np.zeros(N), np.ones(N), np.ones(N), 2 * np.ones(N)))
    
    # Will return an X and y of shapes (5*N, 2) and (5*N)
    # Representing our dataset of 2D samples
    return X, y


X, y = create_data(N)
C = len(np.unique(y))

plt.figure(figsize=(10,8))
plt.plot(X[y==0, 0], X[y==0, 1], 'bx', label="Class 0")
plt.plot(X[y==1, 0], X[y==1, 1], 'ko', label="Class 1");
plt.plot(X[y==2, 0], X[y==2, 1], 'r*', label="Class 2");
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("Data and True Labels")
plt.legend()
plt.show()

#####################################     END Generate Data

# Create coordinate matrices determined by the sample space
xx, yy = np.meshgrid(np.linspace(-4, 4, 250), np.linspace(-4, 4, 250))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.FloatTensor(grid)  











input_dim = X.shape[1]
n_hidden_neurons = 16
output_dim = C







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
num_epochs = 100

# Convert numpy structures to PyTorch tensors, as these are the data types required by the library
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)

# Forward pass through MLP
probs = quick_two_layer_mlp(X_tensor)
print(probs)

# Backpropagation training insert here



from torch.utils.data import DataLoader, TensorDataset

X_test, y_test = create_data(N=100, noise=0.2)

X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# Create your dataset objects
train_data = TensorDataset(X_tensor, y_tensor) 
test_data = TensorDataset(X_test_tensor, y_test_tensor) 

train_dataloader = DataLoader(train_data, batch_size=500, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64) # No need to shuffle...



import time

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
Z_probs = quick_two_layer_mlp(grid_tensor).detach().numpy()
print(Z_probs.shape)
Z_pred = np.argmax(Z_probs, 1).reshape(xx.shape)

fig = plt.figure(figsize=(10,8))

# uses gray background for black dots
plt.pcolormesh(xx, yy, Z_pred, cmap=plt.cm.coolwarm)
    
plt.plot(X_test[y_test==0, 0], X_test[y_test==0, 1], 'bx', label="Class 0")
plt.plot(X_test[y_test==1, 0], X_test[y_test==1, 1], 'ko', label="Class 1");
plt.plot(X_test[y_test==2, 0], X_test[y_test==2, 1], 'r*', label="Class 2");
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("MLP Classification Boundaries Test Set")
plt.legend()
plt.show()



import os

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