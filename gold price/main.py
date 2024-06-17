import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Loading the CSV data into a Pandas DataFrame
gold_data = pd.read_csv('/content/gold price dataset.csv')

# Printing the first 5 rows of the dataframe
print(gold_data.head())

# Printing the last 5 rows of the dataframe
print(gold_data.tail())

# Displaying the number of rows and columns
print(gold_data.shape)

# Getting basic information about the data
print(gold_data.info())

# Checking for missing values
print(gold_data.isnull().sum())

# Getting statistical measures of the data
print(gold_data.describe())

# Calculating correlation matrix
correlation = gold_data.corr()

# Constructing a heatmap to visualize correlations
plt.figure(figsize=(8, 8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')

# Correlation values with respect to 'GLD'
print(correlation['GLD'])

# Checking the distribution of the GLD Price
sns.distplot(gold_data['GLD'], color='green')

# Separating data into features (X) and target (Y)
X = gold_data.drop(['Date', 'GLD'], axis=1)
Y = gold_data['GLD']

# Splitting data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Initializing the RandomForestRegressor model
regressor = RandomForestRegressor(n_estimators=100)

# Training the model
regressor.fit(X_train, Y_train)

# Making predictions on test data
test_data_prediction = regressor.predict(X_test)
print(test_data_prediction)

# Calculating R squared error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error : ", error_score)

# Plotting Actual vs Predicted values
plt.plot(Y_test.values, color='blue', label='Actual Value')
plt.plot(test_data_prediction, color='green', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of samples')
plt.ylabel('GLD Price')
plt.legend()
plt.show()
