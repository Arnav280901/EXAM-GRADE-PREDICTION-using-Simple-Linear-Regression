# importing libraries
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing datasets
X_Train = pd.read_csv('Linear_X_Train.csv')
Y_Train = pd.read_csv('Linear_Y_Train.csv')
X_Test = pd.read_csv('Linear_X_Test.csv')
Y_Test = pd.read_csv('Linear_X_Test.csv')

# training simple linear regression on the training set
regressor = LinearRegression()
regressor.fit(X_Train, Y_Train)

# predicting the test set results
y_pred = regressor.predict(X_Test)

# visualising the training set results
plt.scatter(X_Train, Y_Train, color='red')
plt.plot(X_Train, regressor.predict(X_Train), color='blue')
plt.title('Performance in the exam vs Time spent on coding daily(Training set)')
plt.xlabel('Time')
plt.ylabel('Score')
plt.show()

# visualising the test set results
plt.scatter(X_Test, y_pred, color='red')
plt.plot(X_Test, y_pred, color='blue')
plt.title('Performance in the exam vs Time spent on coding daily(Test set)')
plt.xlabel('Time')
plt.ylabel('Score')
plt.show()
