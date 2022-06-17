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
from sklearn.model_selection import KFold # Important new include

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
np.random.seed(111)
torch.manual_seed(111)

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

N_train = 10000
N_valid = 10000

X_valid, y_valid = generate_data_from_gmm(N_valid, gmm_pdf)

n = X_valid.shape[1]
L = np.array(range(num_classes))
print(L)

# Count up the number of samples per class
N_per_l = np.array([sum(y_valid == l) for l in L])
print(N_per_l)

ax_raw.scatter(X_valid[y_valid == 0, 0], X_valid[y_valid == 0, 1], X_valid[y_valid == 0, 2], c='r', label="Class 0")
ax_raw.scatter(X_valid[y_valid == 1, 0], X_valid[y_valid == 1, 1], X_valid[y_valid == 1, 2], c='b', label="Class 1")
ax_raw.scatter(X_valid[y_valid == 2, 0], X_valid[y_valid == 2, 1], X_valid[y_valid == 2, 2], c='g', label="Class 2")
ax_raw.scatter(X_valid[y_valid == 3, 0], X_valid[y_valid == 3, 1], X_valid[y_valid == 3, 2], c='m', label="Class 3")
ax_raw.set_xlabel(r"$x_1$")
ax_raw.set_ylabel(r"$x_2$")
ax_raw.set_zlabel(r"$x_3$")
# Set equal axes for 3D plots
ax_raw.set_box_aspect((np.ptp(X_valid[:, 0]), np.ptp(X_valid[:, 1]), np.ptp(X_valid[:, 2])))

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
decisions = np.argmax(class_posteriors, axis=0) + 0*np.ones(N_valid) 

# Simply using sklearn confusion matrix
print("\nMAP Results:")
print("Confusion Matrix (rows: Predicted class, columns: True class):")
conf_mat = confusion_matrix(decisions, y_valid)
print(conf_mat)
correct_class_samples = np.sum(np.diag(conf_mat))
print("Total Mumber of Misclassified Samples: {:d}".format(N_valid - correct_class_samples))

# Alternatively work out probability error based on incorrect decisions per class
# perror_per_class = np.array(((conf_mat[1,0]+conf_mat[2,0])/Nl[0], (conf_mat[0,1]+conf_mat[2,1])/Nl[1], (conf_mat[0,2]+conf_mat[1,2])/Nl[2]))
# prob_error = perror_per_class.dot(Nl.T / N)

prob_error = 1 - (correct_class_samples / N_valid)
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
ax_raw.set_xlabel(r"$x_1$")
ax_raw.set_ylabel(r"$x_2$")
ax_raw.set_zlabel(r"$x_3$")
# Set equal axes for 3D plots
ax_raw.set_box_aspect((np.ptp(X_valid[:, 0]), np.ptp(X_valid[:, 1]), np.ptp(X_valid[:, 2])))
plt.title("3D view of MAP Classification Decisions: ")
plt.tight_layout()
plt.show()
map_prob_error=prob_error
###############################################     END Map on Test Data





# Create coordinate matrices determined by the sample space
xx, yy, zz = np.meshgrid(np.linspace(-4, 4, 250), np.linspace(-4, 4, 250), np.linspace(-4, 4, 250))
grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
grid_tensor = torch.FloatTensor(grid)  

input_dim = X_valid.shape[1]
output_dim = len(gmm_pdf['priors'])





#################################################   Define NN and Algos
# The nn.CrossEntropyLoss() loss function automatically performs a log_softmax() to 
# the output when validating, on top of calculating the negative log-likelihood using 
# nn.NLLLoss(), while also being more stable numerically... So don't implement from scratch
criterion = nn.CrossEntropyLoss()
num_epochs = 10    #100


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
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
            
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
    #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    


print("\nCV Running:")
#########################################################   Do CV
def do_cv_on_data(X_train, y_train, X_valid, y_valid):
    nerons = np.arange(10, 55, 5)
    n_nerons= len(nerons)


    # Number of folds for CV
    K = 10

    # STEP 1: Partition the dataset into K approximately-equal-sized partitions
    # Shuffles data before doing the division into folds (not necessary, but a good idea)
    kf = KFold(n_splits=K, shuffle=True) 

    # Allocate space for CV
    # No need for training loss storage too but useful comparison
    mse_valid_mk = np.empty((n_nerons, K)) 
    mse_train_mk = np.empty((n_nerons, K)) # Indexed by model m, data partition k

    # STEP 2: Try all polynomial orders between 1 (best line fit) and 21 (big time overfit) M=2
    nerind=0
    for ner in nerons:
        # K-fold cross validation
        k = 0
        # NOTE that these subsets are of the TRAINING dataset
        # Imagine we don't have enough data available to afford another entirely separate validation set
        for train_indices, valid_indices in kf.split(X_train):
            # Extract the training and validation sets from the K-fold split

            X_train_k = X_train[train_indices]
            y_train_k = y_train[train_indices]
            X_valid_k = X_train[valid_indices]
            y_valid_k = y_train[valid_indices]

            # Convert numpy structures to PyTorch tensors, as these are the data types required by the library
            X_tensor = torch.FloatTensor(X_train_k)
            y_tensor = torch.LongTensor(y_train_k)
            # Backpropagation training insert here

            X_test_tensor = torch.FloatTensor(X_valid_k)
            y_test_tensor = torch.LongTensor(y_valid_k)

            # Create your dataset objects
            train_data = TensorDataset(X_tensor, y_tensor) 
            test_data = TensorDataset(X_test_tensor, y_test_tensor) 

            train_dataloader = DataLoader(train_data, batch_size=500, shuffle=True)
            test_dataloader = DataLoader(test_data, batch_size=64) # No need to shuffle...

            
            # Train model parameters
            quick_two_layer_mlp = nn.Sequential(
                nn.Linear(input_dim, ner),
                nn.ReLU(),
                nn.Linear(ner, output_dim),
                nn.LogSoftmax(dim=1)
            )      

            optimizer = torch.optim.SGD(quick_two_layer_mlp.parameters(), lr=0.01, momentum=0.9, nesterov=True)
            for t in range(num_epochs):
                #print(f"Epoch {t+1}\n-------------------------------")
                model_train_loader(quick_two_layer_mlp, train_dataloader, criterion, optimizer)
                model_test_loader(quick_two_layer_mlp, test_dataloader, criterion)


            # Validation fold polynomial transformation

            # Make predictions on both the training and validation set
            Z_probs = quick_two_layer_mlp(X_test_tensor).detach().numpy()
            Z_pred = np.argmax(Z_probs, 1)
            # Record MSE as well for this model and k-fold
            # Simply using sklearn confusion matrix
            #print("Confusion Matrix (rows: Predicted class, columns: True class):")
            conf_mat = confusion_matrix(Z_pred, y_valid_k)
            #print(conf_mat)
            correct_class_samples = np.sum(np.diag(conf_mat))
            #print("Total Mumber of Misclassified Samples: {:d}".format(len(Z_pred) - correct_class_samples))
            # Alternatively work out probability error based on incorrect decisions per class
            # perror_per_class = np.array(((conf_mat[1,0]+conf_mat[2,0])/Nl[0], (conf_mat[0,1]+conf_mat[2,1])/Nl[1], (conf_mat[0,2]+conf_mat[1,2])/Nl[2]))
            # prob_error = perror_per_class.dot(Nl.T / N)
            prob_error = 1 - (correct_class_samples / len(Z_pred))
            mse_valid_mk[nerind, k] = prob_error


            Z_probs = quick_two_layer_mlp(X_tensor).detach().numpy()
            Z_pred = np.argmax(Z_probs, 1)
            # Record MSE as well for this model and k-fold
            # Simply using sklearn confusion matrix
            #print("Confusion Matrix (rows: Predicted class, columns: True class):")
            conf_mat = confusion_matrix(Z_pred, y_train_k)
            #print(conf_mat)
            correct_class_samples = np.sum(np.diag(conf_mat))
            #print("Total Mumber of Misclassified Samples: {:d}".format(len(Z_pred) - correct_class_samples))
            # Alternatively work out probability error based on incorrect decisions per class
            # perror_per_class = np.array(((conf_mat[1,0]+conf_mat[2,0])/Nl[0], (conf_mat[0,1]+conf_mat[2,1])/Nl[1], (conf_mat[0,2]+conf_mat[1,2])/Nl[2]))
            # prob_error = perror_per_class.dot(Nl.T / N)
            prob_error = 1 - (correct_class_samples / len(Z_pred))
            mse_train_mk[nerind, k] = prob_error

            k += 1

        nerind+=1
                
    # STEP 3: Compute the average MSE loss for that model (based in this case on degree d)
    mse_train_m = np.mean(mse_train_mk, axis=1) # Model average CV loss over folds
    mse_valid_m = np.mean(mse_valid_mk, axis=1) 

    # Plot MSE vs degree
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(nerons, mse_train_m, color="b", marker="s", label=r"$D_{train}$")
    ax.plot(nerons, mse_valid_m, color="r", marker="x", label=r"$D_{valid}$")

    
    ax.legend(loc='upper left', shadow=True)
    plt.xlabel("Number of Neurons")
    plt.ylabel("Probability of error")
    plt.title("Probabilities of error with 5-fold cross-validation for "+str(len(y_train))+" samples")
    plt.show()




    # +1 as the index starts from 0 while the degrees start from 1
    optimal_d = nerons[np.argmin(mse_valid_m)]
    print(mse_valid_mk)
    print("#########################################################################")
    print("The model selected to best fit the data without overfitting is: d={}".format(optimal_d))
    print("#########################################################################")

    # STEP 4: Re-train using your optimally selected model (degree=3) and deploy!!
    # Convert numpy structures to PyTorch tensors, as these are the data types required by the library

    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.LongTensor(y_train)
    # Backpropagation training insert here

    X_test_tensor = torch.FloatTensor(X_valid)
    y_test_tensor = torch.LongTensor(y_valid)

    # Create your dataset objects
    train_data = TensorDataset(X_tensor, y_tensor) 
    test_data = TensorDataset(X_test_tensor, y_test_tensor) 

    train_dataloader = DataLoader(train_data, batch_size=500, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64) # No need to shuffle...


    # Train model parameters
    quick_two_layer_mlp = nn.Sequential(
        nn.Linear(input_dim, optimal_d),
        nn.ReLU(),
        nn.Linear(optimal_d, output_dim),
        nn.LogSoftmax(dim=1)
    )   
    

    optimizer = torch.optim.SGD(quick_two_layer_mlp.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    for t in range(num_epochs):
        #print(f"Epoch {t+1}\n-------------------------------")
        model_train_loader(quick_two_layer_mlp, train_dataloader, criterion, optimizer)
        model_test_loader(quick_two_layer_mlp, test_dataloader, criterion)


    # Validation fold polynomial transformation

    # Make predictions on both the training and validation set
    Z_probs = quick_two_layer_mlp(X_test_tensor).detach().numpy()
    Z_pred = np.argmax(Z_probs, 1)

    # Record MSE as well for this model and k-fold


    # Simply using sklearn confusion matrix
    print("\n\nMLP Results:")
    print("Confusion Matrix (rows: Predicted class, columns: True class):")
    conf_mat = confusion_matrix(Z_pred, y_valid)
    print(conf_mat)
    correct_class_samples = np.sum(np.diag(conf_mat))
    print("Total Mumber of Misclassified Samples: {:d}".format(len(Z_pred) - correct_class_samples))

    # Alternatively work out probability error based on incorrect decisions per class
    # perror_per_class = np.array(((conf_mat[1,0]+conf_mat[2,0])/Nl[0], (conf_mat[0,1]+conf_mat[2,1])/Nl[1], (conf_mat[0,2]+conf_mat[1,2])/Nl[2]))
    # prob_error = perror_per_class.dot(Nl.T / N)

    prob_error = 1 - (correct_class_samples / len(Z_pred))
    print("Probability of Error:"+str(prob_error))
    #########################################################   End CV



    ######################################################## Make Predictions with final trained model

    # Z matrix are the predictions resulting from the forward pass through the network
    Z_probs = quick_two_layer_mlp(X_test_tensor).detach().numpy()
    print(Z_probs.shape)
    Z_pred = np.argmax(Z_probs, 1)

    fig = plt.figure(figsize=(10, 10))
    ax_raw = fig.add_subplot(111, projection='3d')

    # uses gray background for black dots
    #plt.pcolormesh(xx, yy, Z_pred, cmap=plt.cm.coolwarm)
    print("predictions")
    print(Z_pred.shape)
    print(Z_pred)
        
    marker_shapes = '.o^1s+*.o^1s+*'
    marker_colors = 'rbgmgbrrbgmgbr' 
    for r in L: # Each decision option
        for c in L: # Each class label
            ind_rc = np.argwhere((Z_pred==r) & (y_valid==c))

            # Decision = Marker Shape; True Labels = Marker Color
            
            if r == c:
                ax_raw.scatter(X_valid[ind_rc, 0], X_valid[ind_rc, 1], X_valid[ind_rc, 2], c='g', marker = marker_shapes[r])
            else:
                ax_raw.scatter(X_valid[ind_rc, 0], X_valid[ind_rc, 1], X_valid[ind_rc, 2], c='r', marker = marker_shapes[r])



    ax_raw.set_xlabel(r"$x_1$")
    ax_raw.set_ylabel(r"$x_2$")
    ax_raw.set_zlabel(r"$x_3$")
    # Set equal axes for 3D plots
    ax_raw.set_box_aspect((np.ptp(X_valid[:, 0]), np.ptp(X_valid[:, 1]), np.ptp(X_valid[:, 2])))
    plt.title("MLP Classification Predications on Test Set with "+str(len(X_train))+" training samples")
    plt.show()
    return prob_error, optimal_d


############# End of do CV on data function
Training_sizes=[100,200,500,1000,2000,5000]
testing_perfomance=np.zeros(len(Training_sizes))
optimal_ner=np.zeros(len(Training_sizes))
i=0
for T in Training_sizes:
    X_train, y_train = generate_data_from_gmm(T, gmm_pdf)
    print("\n#########################################")
    print("#########################################")
    print("Testing on trainingset size: "+str(T))
    print("#########################################")
    print("#########################################\n")
    testing_perfomance[i], optimal_ner[i] = do_cv_on_data(X_train, y_train, X_valid, y_valid)
    print("\n#########################################")
    print("#########################################")
    print("Finished Testing on trainingset size: "+str(T))
    print("#########################################")
    print("#########################################\n")
    i+=1

print(testing_perfomance)




fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(Training_sizes, testing_perfomance, color="b", marker="s")
plt.axhline(y=map_prob_error, color='r', linestyle='-')
ax.set_xscale('log')

plt.xlabel("Dataset size")
plt.ylabel("Probability of error")
plt.title("Probability of Error Vs. Training Size")
plt.show()




fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(Training_sizes, optimal_ner, color="b", marker="s")
ax.set_xscale('log')

plt.xlabel("Dataset size")
plt.ylabel("Optimal # of Neurons")
plt.title("Optimal Number of Nerons Vs. Training Size")
plt.show()