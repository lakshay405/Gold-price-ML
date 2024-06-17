# Gold-price-ML
Gold Price Prediction using Random Forest Regression
This project focuses on predicting the price of gold using machine learning, specifically Random Forest Regression. The goal is to build a regression model that accurately predicts the price of gold based on historical data and various influencing factors.

Dataset
The dataset (gold price dataset.csv) contains historical data of gold prices along with other relevant features such as crude oil prices, stock prices, and currency exchange rates.

Workflow
Data Loading and Preprocessing:

Load the gold price dataset from a CSV file into a Pandas DataFrame (gold_data).
Display dataset summary including the first and last few rows, dimensions, data types, and check for missing values.
Explore statistical measures of the data and visualize correlations using a heatmap to understand relationships between features.
Exploratory Data Analysis (EDA):

Analyze the distribution of the target variable (GLD, gold price) using a distribution plot (sns.distplot).
Evaluate correlations of all features with respect to the gold price (GLD) to identify influential factors.
Model Training and Evaluation:

Separate the dataset into features (X) and the target variable (Y).
Split the data into training and testing sets using train_test_split.
Initialize and train a Random Forest Regressor model (RandomForestRegressor) with 100 estimators.
Make predictions on the test data and evaluate the model's performance using R squared error (metrics.r2_score).
Visualization:

Plot the actual vs predicted gold prices to visually assess the performance of the regression model.
Libraries Used
numpy and pandas for data manipulation and analysis.
matplotlib and seaborn for data visualization.
sklearn for model selection (RandomForestRegressor), evaluation (train_test_split, metrics.r2_score), and preprocessing.
Conclusion
This project demonstrates the application of Random Forest Regression for predicting gold prices based on historical data and related economic indicators. By leveraging machine learning techniques, the model provides insights into the factors influencing gold prices and achieves accurate predictions, which can be valuable for investors and financial analysts in decision-making processes related to gold investments.
