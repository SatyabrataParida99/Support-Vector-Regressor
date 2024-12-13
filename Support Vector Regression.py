import numpy as np  # For numerical computations
import pandas as pd  # For handling datasets
import matplotlib.pyplot as plt  # For visualizations


data = pd.read_csv(r"D:\FSDS Material\Dataset\Non Linear emp_sal.csv")

# Extract independent variable (x) and dependent variable (y)
x = data.iloc[:, 1:2].values 
y = data.iloc[:, 2].values

# svm model 
from sklearn.svm import SVR
svr_regressor = SVR(kernel='poly',degree=4,gamma='auto')
svr_regressor.fit(x,y)

# Prediction
svr_model_pred = svr_regressor.predict([[6.5]])
print(svr_model_pred)

#Visualizing the SVR results
plt.scatter(x, y, color = 'red')
plt.plot(x, svr_regressor.predict(x), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, svr_regressor.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (SRV)')
plt.xlabel('Position level')
plt.xlabel('Salary')
plt.show()

