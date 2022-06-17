import matplotlib.pyplot as plt # For general plotting

import numpy as np

from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from sklearn.model_selection import KFold

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



alphas=np.geomspace(10**-3,10**3,4)
#alphas=[1]
for alpha in alphas:

    ##################################      Generate data
    N_train=50
    N_test=1000
    n=10
    a=np.random.rand(n)
    mu=np.random.rand(n)
    sigma=np.ones((n,n))+0*np.identity(n)
    print(sigma)


    X_train = multivariate_normal.rvs(mu, sigma, N_train)
    Z_train = multivariate_normal.rvs(np.zeros(n), alpha*np.identity(n), N_train)
    v_train = multivariate_normal.rvs(0, 1, N_train)
    y_train = (a.T.dot((X_train+Z_train).T)+v_train).T

    X_test = np.zeros((n, N_test))
    X_test = multivariate_normal.rvs(mu, sigma, N_test)
    Z_test = multivariate_normal.rvs(np.zeros(n), alpha*np.identity(n), N_test)
    v_test = multivariate_normal.rvs(0, 1, N_test)
    y_test = (a.T.dot((X_test+Z_test).T)+v_test).T

    ##################################      End Generate Data

    x_train = np.column_stack((np.ones(N_train), X_train))  # Prepend column of ones to create augmented inputs x tilde
    X_test = np.column_stack((np.ones(N_test), X_test))  # Prepend column of ones to create augmented inputs x tilde

    def analytical_solution(X, y, beta):
        # Analytical solution is (X^T*X + gamma*ID)^-1 * X^T * y
        n=len(X[0])
        I=np.identity(n)
        return np.linalg.inv(X.T.dot(X)+beta*I).dot(X.T).dot(y)

    def mse(y_preds, y_true):
        # Residual error (X * theta) - y
        error = y_preds - y_true
        # Loss function is MSE
        return np.mean(error ** 2)


    betas=np.geomspace(10**-5,10**5,1000)
    n_betas=len(betas)

    # Number of folds for CV
    K = 10

    # STEP 1: Partition the dataset into K approximately-equal-sized partitions
    # Shuffles data before doing the division into folds (not necessary, but a good idea)
    kf = KFold(n_splits=K, shuffle=True) 

    # Allocate space for CV
    # No need for training loss storage too but useful comparison
    mse_valid_mk = np.empty((n_betas, K)) 
    mse_train_mk = np.empty((n_betas, K)) # Indexed by model m, data partition k

    # STEP 2: Try all polynomial orders between 1 (best line fit) and 21 (big time overfit) M=2
    i=0
    for bet in betas:
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
            
            # Train model parameters
            theta_opt = analytical_solution(X_train_k, y_train_k, bet)

            # Make predictions on both the training and validation set
            y_train_k_pred = X_train_k.dot(theta_opt)     
            y_valid_k_pred = X_valid_k.dot(theta_opt)

            # Record MSE as well for this model and k-fold
            mse_train_mk[i, k] = mse(y_train_k_pred, y_train_k)
            mse_valid_mk[i, k] = mse(y_valid_k_pred, y_valid_k)
            k += 1
        i+=1
                
    # STEP 3: Compute the average MSE loss for that model (based in this case on degree d)
    mse_train_m = np.mean(mse_train_mk, axis=1) # Model average CV loss over folds
    mse_valid_m = np.mean(mse_valid_mk, axis=1) 

    # +1 as the index starts from 0 while the degrees start from 1
    optimal_d = np.argmin(mse_valid_m)
    print("The model selected to best fit the data without overfitting is: d={}".format(optimal_d))

    # STEP 4: Re-train using your optimally selected model (degree=3) and deploy!!
    # ...

    # Plot MSE vs degree
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(betas, mse_train_m, color="b", marker="s", label=r"$D_{train}$")
    ax.plot(betas, mse_valid_m, color="r", marker="x", label=r"$D_{valid}$")

    # Use logarithmic y-scale as MSE values get very large
    ax.set_xscale('log')
    # Force x-axis for degrees to be integer
    #ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend(loc='upper left', shadow=True)
    plt.xlabel("Beta")
    plt.ylabel("MSE")
    plt.title("MSE estimates with {}-fold cross-validation".format(K)+" alpha="+str(alpha))
    plt.show()