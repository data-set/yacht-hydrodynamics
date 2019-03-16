# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importing dataset
column_names = [
        'Longitudinal position of the center of buoyancy',
        'Prismatic coefficient', 
        'Length-displacement ratio', 
        'Beam-draft ratio',
        'Length-beam ratio',
        'Froude number',
        'Residuary resistance per unit weight of displacement'
        ]

dataset = pd.read_csv('yacht_hydrodynamics.csv', sep = ' ',
                      names = column_names)

# Defining independent variable X and dependent variable y
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 6].values


# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
X[:, [1]] = imputer.fit_transform(X[:, [1]])


# Splitting data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
                                                    random_state = 0)


# Creating a random forest model and fitting the training set
from sklearn.ensemble import RandomForestRegressor
rfr_model = RandomForestRegressor()
rfr_model.fit(X_train, y_train)


# Predicting test set results
y_pred = rfr_model.predict(X_test)


# Calculating mean absolute error
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
# mae output is non-negative floating point. The best value is 0.0.
print("Mean absolute error = {}".format(mae))


# Visualising residuary resistance against individual features (all plots)
for i in range(6):
    plt.subplot(2, 3, i + 1)    
    plt.scatter(X[:, i], y, color = 'blue', marker = '.')
    plt.xlabel(column_names[i])
    plt.ylabel("Residuary resistance")
    plt.show()
    

# Uncomment following lines to get each plot separately

#plt.scatter(X[:, 0], y, color = 'blue', marker = '.')
#plt.xlabel(column_names[0])
#plt.ylabel("Residuary resistance")
#plt.show()
#
#plt.scatter(X[:, 1], y, color = 'blue', marker = '.')
#plt.xlabel(column_names[0])
#plt.ylabel("Residuary resistance")
#plt.show()
#
#plt.scatter(X[:, 2], y, color = 'blue', marker = '.')
#plt.xlabel(column_names[0])
#plt.ylabel("Residuary resistance")
#plt.show()
#
#plt.scatter(X[:, 3], y, color = 'blue', marker = '.')
#plt.xlabel(column_names[0])
#plt.ylabel("Residuary resistance")
#plt.show()
#
#plt.scatter(X[:, 4], y, color = 'blue', marker = '.')
#plt.xlabel(column_names[0])
#plt.ylabel("Residuary resistance")
#plt.show()
#
#plt.scatter(X[:, 5], y, color = 'blue', marker = '.')
#plt.xlabel(column_names[0])
#plt.ylabel("Residuary resistance")
#plt.show()    
