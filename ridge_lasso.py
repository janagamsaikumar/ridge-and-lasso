# imported the libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

#Importing Linear Regression Machine Learning Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
dataset=pd.read_csv(r'C:\Users\saikumar\Desktop\AMXWAM data science\AMXWAM_ TASK\TASK-30\car-mpg (1).csv')
h=pd.DataFrame(dataset.head(10))
t=pd.DataFrame(dataset.tail())
des=dataset.describe() 
dataset.info()
# make it clear keep numerical to prediction no catergorical data
# which leads to over fitting problem 
 
dataset=dataset.drop(columns='car_name',axis=1)
dataset.info()
# dropped off the car names 
# we are going to replace origin data by mentioning with country names as we observed they classified as 3 
# we changed each one belong one country
dataset['origin']=dataset['origin'].replace({1:'asia',2:'uk',3:'america'})
dataset=pd.get_dummies(dataset,columns=['origin'])
dataset.isnull().sum()
q=(dataset=='?').sum() # we have 6 '?' in hp
dataset = dataset.replace('?', np.nan)
nann=(dataset=='NaN').sum()

# treating our missing data using EDA techniques
dataset = dataset.apply(lambda x: x.fillna(x.median()), axis = 0)
dataset.isnull().sum()

# depdendent and independent varibles for model building 
X = dataset.drop(['mpg'], axis = 1).values # independent variable
y = dataset[['mpg']].values #dependent variable

# go for train and test data 
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size = 0.30, random_state = 1)
X_train.shape

# we have to do feature scaling 
# if no featurescaling one column dominates the other
sc=StandardScaler() 
X_train=sc.fit_transform(X_train)
y_train=sc.fit_transform(y_train)
X_test=sc.fit_transform(X_test)
y_test=sc.fit_transform(y_test)




# simple linear model
reg = LinearRegression()
reg.fit(X_train, y_train)

# finding the coefficient
# for idx, col_name in enumerate(X_train.columns):
#     print('The coefficient for {} is {}'.format(col_name, reg.coef_[0][idx]))
    
# intercept = reg.intercept_[0]
# print('The intercept is {}'.format(intercept))

# ridge regression
ridge_reg=Ridge(alpha = 0.3)
ridge_reg.fit(X_train, y_train)

# lasso regresison 
lasso_model = Lasso(alpha = 0.1)
lasso_model.fit(X_train, y_train)

# score compariison

#Simple Linear Model
print(reg.score(X_train, y_train)) # 83%
print(reg.score(X_test, y_test))   # 85% 

print('*************************')
#Ridge
print(ridge_reg.score(X_train, y_train)) # 83%
print(ridge_reg.score(X_test, y_test))   # 85%

print('*************************')
#Lasso
print(lasso_model.score(X_train, y_train)) # 79%
print(lasso_model.score(X_test, y_test))  #83%

## model parameter tuning ##
# increasing the model performance without high variance or over fitting 
# when we cannot observe in r-square as it increases no. of attributes yet no improvement 
# we go with adjacent r-square which statistically improves the r-square

# data_train_test = pd.concat([X_train, y_train], axis=1)
# data_train_test.head()

# lets check the sse=sum of square error= Actual(a value to be predicted)-predicted(predicted values )
mse=np.mean((reg.predict(X_test)-y_test)**2)
# it has given 0.14 which is the average variance between actual and predicted values
# mse is the standard deviation we know standard deviation sigma =square root of variance 
import math as m
sigma_mse=m.sqrt(mse)
print('Root Mean Squared Error: {}'.format(sigma_mse)) #0.377 is the diff between actual and predicted 

# Is OLS a good model ? Lets check the residuals for some of these predictor.

# fig.set_size_inches(10,8,forward=True)
# sns.residplot(x= X_test['hp'], y= y_test['mpg'], color='green', lowess=True )

y_pred = reg.predict(X_test)

# Since this is regression, plot the predicted y value vs actual y values for the test data
# A good model's prediction will be close to actual leading to high R and R2 values
#plt.rcParams['figure.dpi'] = 500
plt.scatter(y_test, y_pred)
# we observe a complete positive correlation between my actual and preidicted data
# my data points are properly spread in positive direction hence saying strong correlation between attributes
# we can consider now based on the graph 
# ridge regression gave a better score compared to lasso 

# ridge 
y_pred_ridge = ridge_reg.predict(X_test)
plt.scatter(y_test, y_pred_ridge)
#lasso
y_pred_lasso = lasso_model.predict(X_test)
plt.scatter(y_test, y_pred_lasso)