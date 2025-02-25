import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
#import joblib
#import neptune
#from neptune.types import File
import os
import matplotlib.pyplot as plt

lr = LinearRegression()
########################################################################
####
# Load and split data
for _ in range(10):
    rng = np.random.RandomState(_)
    x = 10 * rng.rand(1000).reshape(-1,1)
    y = 2 * x - 5 + rng.randn(1000).reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=70)
    ########################################################################
    ######
    # Fitting the model
    lr.fit(X_train, y_train)
    y_preds = lr.predict(X_test)
    test_mse = mean_squared_error(y_test, y_preds )
    average_mse = np.mean(test_mse)
    print(f'MSE Result: { test_mse}')

print("Average Mean Squared Error:", average_mse)
with open('metrics.txt', 'w') as outfile:
    outfile.write(f'Mean Squared Error = {average_mse}.')
# Plotting the training data
plt.scatter(X_train, y_train, color='blue', label='Training data')
# Plotting the testing data
plt.scatter(X_test, y_test, color='red', label='Testing data')
plt.scatter(X_test, y_preds, c="g", label="Predictions")
# Show the legend
plt.xlabel('X')
plt.ylabel('y')
plt.title('Training and Testing Data Split')
plt.legend()
plt.grid(True)
plt.savefig('model_results.png', dpi=120)
