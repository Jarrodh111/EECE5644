import matplotlib.pyplot as plt # For general plotting

import numpy as np

import hw2q2
from scipy.optimize import minimize
from scipy.stats import multivariate_normal # MVN not univariate

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

################################## Function to generate data
def hw2q2():
    Ntrain = 100
    data = generateData(Ntrain)
    plot3(data[:, 0], data[:, 1], data[:, 2], name="Training")
    xTrain = data[:, 0:2]
    yTrain = data[:, 2]

    Ntrain = 1000
    data = generateData(Ntrain)
    plot3(data[:, 0], data[:, 1], data[:, 2], name="Validation")
    xValidate = data[:, 0:2]
    yValidate = data[:, 2]
    return xTrain, yTrain, xValidate, yValidate


def generateData(N):
    gmmParameters = {}
    gmmParameters['priors'] = [.3, .4, .3]  # priors should be a row vector
    gmmParameters['meanVectors'] = np.array([[-10, 0, 10], [0, 0, 0], [10, 0, -10]])
    gmmParameters['covMatrices'] = np.zeros((3, 3, 3))
    gmmParameters['covMatrices'][:, :, 0] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    gmmParameters['covMatrices'][:, :, 1] = np.array([[8, 0, 0], [0, .5, 0], [0, 0, .5]])
    gmmParameters['covMatrices'][:, :, 2] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    X = generateDataFromGMM(N, gmmParameters)
    return X


def generateDataFromGMM(N, gmmParameters):
    #    Generates N vector samples from the specified mixture of Gaussians
    #    Returns samples and their component labels
    #    Data dimensionality is determined by the size of mu/Sigma parameters
    priors = gmmParameters['priors']  # priors should be a row vector
    meanVectors = gmmParameters['meanVectors']
    covMatrices = gmmParameters['covMatrices']
    n = meanVectors.shape[0]  # Data dimensionality
    C = len(priors)  # Number of components
    X = np.zeros((n, N))
    labels = np.zeros((1, N))
    # Decide randomly which samples will come from each component
    u = np.random.random((1, N))
    thresholds = np.zeros((1, C + 1))
    thresholds[:, 0:C] = np.cumsum(priors)
    thresholds[:, C] = 1
    for l in range(C):
        indl = np.where(u <= float(thresholds[:, l]))
        Nl = len(indl[1])
        labels[indl] = (l + 1) * 1
        u[indl] = 1.1
        X[:, indl[1]] = np.transpose(np.random.multivariate_normal(meanVectors[:, l], covMatrices[:, :, l], Nl))

    # NOTE TRANPOSE TO GO TO SHAPE (N, n)
    return X.transpose()


def plot3(a, b, c, name="Training", mark="o", col="b"):
    # Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(a, b, c, marker=mark, color=col)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$y$")
    plt.title("{} Dataset".format(name))
    # To set the axes equal for a 3D plot
    ax.set_box_aspect((np.ptp(a), np.ptp(b), np.ptp(c)))
    plt.show()
##################### End Function to generate data




########################################### Generate Data
xTrain, yTrain, xValidate, yValidate = hw2q2()
NTrain=len(xTrain)
xTrain = np.column_stack((np.ones(NTrain), xTrain))  # Prepend column of ones to create augmented inputs x tilde
X=xTrain
y=yTrain
N=NTrain
########################################### End Generate Data



########################################### Transform Data
def cubic_transformation(X):
    n = X.shape[1]
    phi_X = X
    
    # Take all monic polynomials for a quadratic
    phi_X = np.column_stack((phi_X,
                             X[:, 1] * X[:, 1],
                             X[:, 1] * X[:, 2],
                             X[:, 2] * X[:, 2],
                             X[:, 1] * X[:, 1] * X[:, 1],
                             X[:, 1] * X[:, 1] * X[:, 2],
                             X[:, 1] * X[:, 2] * X[:, 2],
                             X[:, 2] * X[:, 2] * X[:, 2]))
        
    return phi_X
X=cubic_transformation(X)
########################################### End Transform Data


def analytical_solution(X, y, gamma):
    # Analytical solution is (X^T*X + gamma*ID)^-1 * X^T * y
    n=len(X[0])
    I=np.identity(n)
    return np.linalg.inv(X.T.dot(X)+gamma*I).dot(X.T).dot(y)

gamma=0
theta_opt = analytical_solution(X, y, gamma)
analytical_preds = X.dot(theta_opt)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 1], X[:, 2], y, label="Training Data")

# Plot the OLS regression line on our original scatter plot
ax.scatter(X[:, 1], X[:, 2], analytical_preds, color='k', label="OPT")

ax.legend()
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_zlabel(r"$y$")
ax.set_box_aspect((np.ptp(X[:, 1]), np.ptp(X[:, 2]), np.ptp(analytical_preds)))
plt.title("Testing on Training Dataset")
plt.show()

###################################### Test on validation set
N_test = len(xValidate)

X_test = xValidate
X_test = np.column_stack((np.ones(N_test), X_test))  # Prepend column of ones to create augmented inputs x tilde

# y = X*theta_true + noise
y_test = yValidate

fig = plt.figure(figsize=(12, 10))
ax_test = fig.add_subplot(111, projection='3d')
ax_test.scatter(X_test[:, 1], X_test[:, 2], y_test, label="True Data")

# Predictions with our MLE theta
X_test=cubic_transformation(X_test)
analytical_preds = X_test.dot(theta_opt)

# Plot the OLS regression line on our original scatter plot
ax_test.scatter(X_test[:, 1], X_test[:, 2], analytical_preds, color='k', label="Predictions")

# Plot the learned regression line on our original scatter plot AND the new unseen data
ax_test.legend()
ax_test.set_xlabel(r"$x_1$")
ax_test.set_ylabel(r"$x_2$")
ax_test.set_zlabel(r"$y$")
ax_test.set_box_aspect((np.ptp(X_test[:, 1]), np.ptp(X_test[:, 2]), np.ptp(analytical_preds)))
plt.title("Testing on Validation Dataset")
plt.show()

######################## find mean squared error
mse=0
for i in range(N_test):
    er=(y_test[i]-analytical_preds[i])**2
    mse+=er
mse=mse/N_test
print("Mean Squared Error for MLE Case: "+str(mse))


test_gammas= np.geomspace(10**-4,10**4,1000)
gamma_mse=np.zeros(len(test_gammas))
j=0
for g in test_gammas:
    theta_opt = analytical_solution(X, y, g)
    analytical_preds = X_test.dot(theta_opt)
    mse=0
    for i in range(N_test):
        er=(y_test[i]-analytical_preds[i])**2
        mse+=er
    mse=mse/N_test
    gamma_mse[j]=mse
    j+=1

fig, ax = plt.subplots()
plt.plot(test_gammas,gamma_mse,'b.')
ax.set_xscale('log')
plt.xlabel("Gamma Values")
plt.ylabel("Mean Squared Error")
plt.title("MSE vs Gamma")
plt.grid(True)
plt.show()