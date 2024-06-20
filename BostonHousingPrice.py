#!/usr/bin/env python
# coding: utf-8

# In[321]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# 
# ## Loading the data set
# 

# In[322]:


df = pd.read_csv(r'C:\Users\Muzhg\OneDrive\Desktop\housing.csv') 


# In[323]:


df.head()


# 
# ## Missing Data Analysis
# 

# In[324]:


# Check for missing values
missing_values = df.isnull().sum()

# Calculate the percentage of missing data in each column
missing_percentage = (missing_values / len(df)) * 100

# Display the missing data statistics
print("Missing Values in Each Column:\n", missing_values)
print("\nPercentage of Missing Data:\n", missing_percentage)


# In[325]:


# Remove rows with missing values
df.dropna(inplace=True)

# Verify that missing values have been removed
print("\nMissing values in each column after removal:")
print(df.isnull().sum())


# 
# ## Data Exploration and Visualization
# 

# In[326]:


sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(df['median_house_value'], color='purple', kde=True)
plt.title('Distribution of Median House Values')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.show()


# ## Finding and removing the outliars

# In[327]:


# Assuming 'data' is your DataFrame and 'median_house_value' is the column of interest
Q1 = df['median_house_value'].quantile(0.25)
print(Q1)
Q3 = df['median_house_value'].quantile(0.75)
print(Q3)
IQR = Q3 - Q1


# Define the bounds for the outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
data_no_outliers = df[(df['median_house_value'] >= lower_bound) & (df['median_house_value'] <= upper_bound)]

# Check the shape of the data before and after removal of outliers
print("Original data shape:", df.shape)
print("New data shape without outliers:", data_no_outliers.shape)


# In[328]:


df1=data_no_outliers


# 
# ## Checking for correlations between independent variables
# 

# In[329]:


plt.figure(figsize=(12, 8))
sns.heatmap(df1.corr(), annot=True, cmap='Purples')
plt.title('Correlation Heatmap of Housing Data')
plt.show()


# In[330]:


df1 = df1.drop("total_bedrooms", axis = 1)
df1.columns


# 
# ## String Data Categorization to Dummy Variables
# 

# In[331]:


ocean_proximity_dummies = pd.get_dummies(df1['ocean_proximity'], prefix='ocean_proximity')
df1 = pd.concat([df1.drop("ocean_proximity", axis =1), ocean_proximity_dummies], axis=1)
ocean_proximity_dummies


# In[332]:


df1 = df1.drop("ocean_proximity_ISLAND", axis = 1)
df1.columns


# 
# ## Splitting the Data into Train/Test
# 

# In[333]:


#Define the features (independent variables) and target (dependent variable)
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'population', 'households', 'median_income',
       'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND',
       'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN']
target = ["median_house_value"]

X = df1[features]
y = df1[target]


# Split the data into a training set and a testing set
# test_size specifies the proportion of the data to be included in the test split
# random_state ensures reproducibility of your split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1111)

# Check the size of the splits
print(f'Training set size: {X_train.shape[0]} samples')
print(f'Test set size: {X_test.shape[0]} samples')


# 
# Training 
# 

# In[334]:


X_train


# In[335]:


# Adding a constant to the predictors because statsmodels' OLS doesn't include it by default
X_train_const = sm.add_constant(X_train)
X_train_const


# In[336]:


# Fit the OLS model
model_fitted = sm.OLS(y_train, X_train_const).fit()

# Printing Summary
print(model_fitted.summary())


# 
# ## Prediction/Testing
# 

# In[337]:


# Adding a constant to the test predictors
X_test_const = sm.add_constant(X_test)

# Making predictions on the test set
test_predictions = model_fitted.predict(X_test_const)
test_predictions


# In[338]:


X_test_const


# In[339]:


y_test


# In[340]:


y_test = y_test.values.flatten()  # Convert DataFrame to a 1D numpy array
test_predictions = test_predictions.values  # Convert Series to a numpy arra


# 
# ## Checking OLS Assumtions
# 

# 
# ## Assumtion 1: Linearity
# 

# In[341]:


# Scatter plot for observed vs predicted values on test data
plt.scatter(y_test, test_predictions, color = "purple")
plt.xlabel('Observed Values')
plt.ylabel('Predicted Values')
plt.title('Observed vs Predicted Values on Test Data')
plt.plot(y_test, y_test, color='darkred')
plt.show()



# Positive Linear Relationship: The red line (which represents a perfect prediction line) 
# and the distribution of the data points suggest there's a positive linear relationship 
# between the observed and predicted values. This means that as the actual values increase, 
# the predicted values also increase, which is a good sign for linearity.

# 
#  ## Assumtion 2: Random Sample
# 

# In[342]:


# Calculate the mean of the residuals
mean_residuals = np.mean(model_fitted.resid)

print(f"The mean of the residuals is {np.round(mean_residuals,2)}")


# In[343]:


# Plotting the residuals
plt.scatter(model_fitted.fittedvalues, model_fitted.resid, color = "pink")
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()


# In this plot, there are no noticeable patterns. The residuals are scattered randomly 
# around the horizontal line at zero, without any distinct shape or trend.
# If a pattern were present, or if the residuals consistently deviated 
# from zero, it might indicate problems like model misspecification,
# non-linearity, or missing important variables

# ## Assumtion 3: Exogeneity

# In[344]:


# Calculate the residuals
residuals = model_fitted.resid

# Check for correlation between residuals and each predictor
for column in X_train.columns:
    corr_coefficient = np.corrcoef(X_train[column], residuals)[0, 1]
    print(f'Correlation between residuals and {column}: {np.round(corr_coefficient,2)}')


# Durbin-Wu-Hausman Test: To conduct a more formal statistical test,
# you can use the Durbin-Wu-Hausman test. This test compares your current model to an
# alternative model that includes an instrumental variable. It evaluates whether 
# the coefficients of the original model significantly change when the potentially
# endogenous variables are replaced with their instrumented values. This advanced
# econometric approach helps determine the presence of endogeneity, but it requires 
# the identification of suitable instruments, which can be challenging.

# ## Assumtion 4: Homoskedasticty

# In[345]:


# Plotting the residuals
plt.scatter(model_fitted.fittedvalues, model_fitted.resid, color = "pink")
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()


# Random Scatter:
# If the plot shows a random scatter of residuals around the horizontal line at zero, it supports 
# the assumption of homoscedasticity, indicating that the variance of the residuals is 
# constant across all levels of the independent variables.
# 
# Pattern or Shape:
# If the residuals display a distinct pattern, such as a curve, or form a funnel shape where the 
# spread increases with the fitted values, it suggests heteroscedasticity. 
# This means that the variance of the residuals changes with the level of the independent variables.

# In[346]:


# I want to see how exactly my model looks like.
intercept = model_fitted.params['const']
coefficients = model_fitted.params[1:]

model_formula = f"y = {intercept:.4f} + " + " + ".join([f"{coeff:.4f}*{name}" for name, coeff in coefficients.items()])
print("\nActual Model Formula:")
print(model_formula)


# In[ ]:




