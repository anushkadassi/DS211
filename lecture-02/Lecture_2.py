import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# use pandas to load real_estate_dataset.csv 
df = pd.read_csv("real_estate_dataset.csv")

# get the number of samples and features
n_samples, n_features = df.shape 

print(f"Number of samples and features: {n_samples, n_features}")

# get the names of the columns 
columns = df.columns 

# save the column names to a file for accessing later as text file 
np.savetxt("column_names.txt", columns, fmt = "%s")

# use Square_feet, Garage_size, Location-score, Distance_to_center as features 
X = df[["Square_Feet" , "Garage_Size", "Location_Score", "Distance_to_Center"]]

# Use price as the target 
y = df["Price"].values 

print(f"Shape of X: {X.shape}\n")
print(f"data type of X : {X.dtypes}\n")

# get the number of sample and features in X
n_samples, n_features = X.shape 

# bias is the expected value of the target variable 

# Build a linear model to predict price from the four features in X
# make an array of coefficients of the size of n_features + 1. Initisalise to 1

coefs = np.ones(n_features + 1)

# predict the price for each sample in X 
prediction_bydefn = X @ coefs[1:] + coefs[0]

# append a column of 1s to X 
X = np.hstack((np.ones((n_samples, 1)), X))

# predict the prize for each sample in X 
predictions = X @ coefs

# See if all entries in predbydef and predictions are the same 
is_same = np.allclose(prediction_bydefn, predictions )

print(f"Are the predictions the same with X*coefs[1:] + coefs[0] and X*coefs? {is_same}")

# calculate the error using the prediction and y 
errors = y - predictions

# calculate the relative error(should be less than 1)
rel_errors = errors / y

# calculate the mean of square of errors using a loop 
loss_loop = 0 
for i in range(n_samples):
    loss_loop += errors[i]**2
loss_loop = loss_loop/n_samples  

# calculate the mean of square of errors using the matrix operation 
loss_matrix = np.transpose(errors) @ errors/ n_samples

# compare the two methods of calulating mean squared errors 
is_diff = np.allclose(loss_loop, loss_matrix)
print(f'Are the loss by direct and matrix same? {is_diff}\n' )

# print the size of errors and its L2 norm 
print(f"Size of errors : {errors.shape}")
print(f"L2 norm of errors: {np.linalg.norm(errors)}")
print(f"L2 norm of relative errors: {np.linalg.norm(rel_errors) }")

# What is the optimisation problem? 
# I want to find the coefficients that minimise the mean squared error 
# this problem is called as least squares problem 

# aside 
# In Nu = f(Re, Pr); Nu = \alpha*Re^m*Pr^n, we want to find \alpha, m and n that minimise the error 
# Pr is Prandtl number, Re is Reynolds number, Nu is Nusselt number
# Objective function: f(coefs) = 1/n_samples * \sum_{i = 1}^{n_samples} {y_i - \sum_{j=1}^ {n_features + 1}}

# What is a solution? 
# A solution is a set of coefficients that minimize the objective function 

# How do I find a solution? 
# By searching for the coefficients for which the gradient of the objective function is zero 
# Or I can set the gradient of the objective function to zero and solve for the 

# write the loss_matrix in terms of the data and coefficients 
loss_matrix = (y - X@coefs).T @ (y - X @coefs) / n_samples 

# Calculate the gradient of the loss with respect to the coefficients 
grad_matrix = -2/n_samples * X.T @ (y - X @ coefs)

# We see grad_matrix = 0 and solve for coefs
# X.T @ y = X.T @ X @ coefs 
# X.T @ X @ coefs = X.T @ y. This equation is called the Normal Equation 
# coefs = (X.T @ X) ^ {-1} @ X.T @ y

coefs = np.linalg.inv(X.T @ X) @ X.T @ y 

# save coefs to a file for viewing 
np.savetxt("coefs.csv", coefs, delimiter= ",")

# calculate the prediction using the optimal coefficients 
prediction_model = X @ coefs 

# calculate the errors using the optimal coefficients 
errors_model =  y - prediction_model

rel_errors_model = errors_model / y

# print the L2 norm of the errors_model 
print(f"L2 norm of the errors_model: {np.linalg.norm(errors_model)}")

#print the L2 norm of the relative errors_model
print(f"L2 norm of relative errors_model: {np.linalg.norm(rel_errors_model)}")

# Use all the  features in the dataset to build a linear model to predict Price 
X = df.drop("Price", axis = 1).values 
y = df["Price"].values 

# get the number of samples and features in X 
n_samples, n_features = X.shape 

# To solve the linear model using the normal equations 
X = np.hstack((np.ones((n_samples, 1)), X))
coefs = np.linalg.inv(X.T @ X) @ X.T @ y


# Save coefs to a file named coefs_all.csv
np.savetxt("coefs_all.csv", coefs, delimiter = ",")

# Calculate the rank of X.T @ X, to know if unique solution exists 
rank_XTX = np.linalg.matrix_rank(X.T @ X)

print(f"Rank of X.T @ X: {rank_XTX}")
# Do QR factorisation 
# Solve the normal equation using matrix decomposition
# This will decompose X.T @ X = QR, where Q is an orthogonal matrix and R is an upper triangular matrix
# We are doing this because this is often more stable than an inverse 
# Q and R are (n_features+1) × (n_features+1)

b = X.T @ y

Q, R = np.linalg.qr(X.T @ X)

# write R to a file named R.csv 
np.savetxt("R.csv", R, delimiter = ",")

# R * coefs = b
sol = Q.T @ Q
np.savetxt("sol.csv", sol, delimiter = ",")

print(f"Shape of y: {y.shape}")
print(f"Shape of Q: {Q.shape}")

b = Q.T @ b
coefs_qr = np.linalg.inv(R) @ b

# loop to solve R * coefs = b using back substitution 
coefs_qr_loop = np.zeros(n_features + 1)
for i in range(n_features, -1, -1):
    coefs_qr_loop[i] = b[i]
    for j in range(i + 1, n_features + 1):
        coefs_qr_loop[i] -= R[i, j] * coefs_qr_loop[j]
    coefs_qr_loop[i] = coefs_qr_loop[i]/R[i, i]

# save coefs_qr_loop to a file named coefs_qr_loop.csv
np.savetxt("coefs_qr_loop.csv", coefs_qr_loop, delimiter = ",")

# Eigen decomposition of a square matrix 

# A = VDV^T

# Calculate coef_svd using the pseudoinverse of X

# S is a vector of singular values of X
# U is a matrix of left singular vectors of X
# Vt is a matrix of right singular vectors of X
U, S, Vt = np.linalg.svd(X, full_matrices = False)

np.savetxt("U.csv", U, delimiter = ",")
np.savetxt("S.csv", S, delimiter = ",")
np.savetxt("Vt.csv", Vt, delimiter = ",")

# Creating diaognal matrix D from S
D = np.diag(S)

# Using reciprocals of the singular values we create the pseudoinverse of X
X_pinv = Vt.T @ np.linalg.inv(D) @ U.T

# Calculate coefs_svd using the pseudoinverse of X
coefs_svd = X_pinv @ y

# Save coefs_svd to a file named coefs_svd.csv
np.savetxt("coefs_svd.csv", coefs_svd, delimiter = ",")


# write X as a product of U, S and Vt
X_svd = U @ np.diag(S) @ Vt

# Solve for X_svd @ coeffs_svd = y

# Normal equation : X_svd^T @ X_svd @ coefs_svd = X_svd^T @ y
# replace X_svd with U @ np.diag(S) @ Vt
# Vt^T @ np.diag(S)^2 @ Vt @ coefs_svd = Vt^T @ np.diag(S) @ U^T @ y
# np.diag(S)^2 @ Vt @ coefs_svd = np.diag(S) @ U^T @ y
# coefs_svd = Vt @ np.diag(S)^{-1} @ U^T @ y

coef_svd = Vt.T @ np.diag(1/S) @ U.T @ y    # 1/S is faster but may fail first if S has zeros
coefs_svd_pinv = np.linalg.pinv(X) @ y

# Save coefs_svd to a file named coefs_svd.csv
np.savetxt("coefs_svd_pinv.csv", coefs_svd_pinv, delimiter = ",")
# Save coefs_svd_pinv to a file named coefs_svd_pinv.csv
np.savetxt("coefs_svd_pinv.csv", coefs_svd_pinv, delimiter = ",")


#X_1 = X[:,1]
#coeffs_1 = np.linalg.inv(X_1.T @ X_1) @ X_1.T @ y



# plot the data on X[:, 1] vs y axis 
# Also plot a regression line with only X[:,0] and X[:,1] as features
# first make X[:, 1] as np.arange between min and max of X[:,1]
# then calculate the predictions using the coefficients 
#X_feature = np.arange(np.min(X_1[:, 0]), np.max(X_1[:, 0]), 0.01)
#plt.scatter(X[:, 0], y)
#plt.plot(X_feature , X_feature *  coeffs_1, color = "red")   # coefficients were calculated using all the features 
#plt.plot()
#plt.xlabel("Square Feet")
#plt.ylabel("Price")
#plt.title("Price vs Square Feet")
#plt.show()
#plt.savefig("Price_vs_Square_Feet.png")

# Orthogonal regression fit 


# Use X as only the square feet to build a lineae model to predict Price 
X = df["Square_Feet"].values.reshape(-1, 1)
y = df["Price"].values

# add a column of 1s to X 
X = np.hstack((np.ones((n_samples, 1)), X))

coeffs = np.linalg.inv(X.T @ X) @ X.T @ y


# Create points for the regression line
X_feature = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100).reshape(-1, 1)
X_feature = np.hstack((np.ones((100, 1)), X_feature))

# X_feature = np.arange(np.min(X[:, 1]), np.max(X[:, 1]), 10).reshape(-1, 1)

print(f"min of X[:, 1]: {np.min(X[:, 1])}")
print(f"max of X[:, 1]: {np.max(X[:, 1])}")

plt.scatter(X[:, 1], y, color = 'blue')
plt.plot(X_feature[:, 1], X_feature @ coeffs , color = 'red')
plt.xlabel("Square Feet")
plt.ylabel("Price")
plt.title("Price vs Square Feet")
plt.show()




