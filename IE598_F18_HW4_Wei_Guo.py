#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 11:07:05 2018

@author: guowei
"""

#import all pacekages used in this program
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets.samples_generator import make_regression
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

#data generating...
samples_No = 1000
features_No = 100
X, y, cf= make_regression(n_samples = samples_No, n_features = features_No, 
                          noise = 0, coef = True, random_state = 42)

#name each features from feature1 to feature100
col_names = []
for i in range(1,features_No + 1):
    col_names.append('Feature'+str(i))

#put the features and target into a dataframe
data_feature = pd.DataFrame(X,columns = col_names)
data_result = pd.DataFrame({'Target':y})
data = pd.concat([data_feature,data_result],axis = 1)

#calculate the missing rate of each column
missRate = data.apply(lambda x: (len(x)-x.count())/len(x)*100)
missRate = pd.DataFrame(missRate)
missRate = missRate.rename(columns={0:'Miss_Rate'})

#get basic statistics info of each column
data_stat = data.describe().T

#get types of each column
data_type = pd.DataFrame(data.dtypes)
data_type = data_type.rename(columns={0:'Type'})

#merge the missing rate, basic statistics and types into a dataframe
data_desc = missRate.merge(data_type,how = 'left', left_index = True, right_index = True) \
                .merge(data_stat,how = 'left', left_index = True, right_index = True)

#draw a correlation coefficient matrix with a heatmap
Corr = data.corr()
hm = sns.heatmap(Corr,cbar = True,cmap="YlGnBu")
plt.title('heatmap of correlation coefficient matrix')
plt.show()

#split the data into trainning and testing set
X = data.iloc[:,0:-1]
y = data.iloc[:,-1]
X = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

#First linear regression model, with all 100 features
slr = LinearRegression()
slr.fit(X_train, y_train)

y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

#plot residual errors of LR model1
plt.scatter(y_train_pred,y_train_pred - y_train, 
            c = 'steelblue',marker = 'o',
            edgecolor = 'white',
            label = 'Trainning data')

plt.scatter(y_test_pred,y_test_pred - y_test, 
            c = 'limegreen',marker = 's',
            edgecolor = 'white',
            label = 'Test data')

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('residual errors of LR model1')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = y.min(), xmax = y.max(), color = 'black',lw = 2)
plt.xlim = ([y.min(),y.max()])
plt.show()

#print performance indicators
print("MSE train: ")
print(round(mean_squared_error(y_train,y_train_pred),10))
print("MSE test: ")
print(round(mean_squared_error(y_test,y_test_pred),10))

print("R^2 train: ")
print(round(r2_score(y_train,y_train_pred),10))
print("R^2 test: ")
print(round(r2_score(y_test,y_test_pred),10))

#print model coefficients and intercept
print('coefficients:')
print(slr.coef_.tolist())
print('intercept:')
print(slr.intercept_.tolist())

coeff_lr_1 = pd.DataFrame({'coeff':slr.coef_.tolist()})
coeff_lr_1.to_excel('coeff_lr_1.xlsx')





#pick features that are significantly correlated with target
#threshold is set to be 0.1
Threshold = 0.10
Corr_result = Corr[['Target']]
Corr_result = Corr_result[(Corr_result['Target'] >= Threshold) | 
        (Corr_result['Target'] <= -Threshold)]
feature_in_use = Corr_result.index.tolist()
data_for_model = data[feature_in_use]

#draw a scatterplot matrix of remaining features
sns.pairplot(data_for_model[feature_in_use],size = 2.5)
plt.tight_layout()
plt.show()

#draw a correlation coefficient matrix with a heatmap
cm = data_for_model.corr()
plt.subplots(figsize=(9, 9))
plt.title('Matrix\n')
hm = sns.heatmap(cm,cbar = True,annot = True,square = True,
                 fmt = '.2f',annot_kws = {'size':15},cmap="YlGnBu")
plt.show()

#draw a boxplot of the features
box = data_for_model.iloc[:,0:-1].values
plt.boxplot(box,notch=False, sym='rs',vert=True)
plt.title('Features box plot')
plt.xticks([y+1 for y in range(8)], 
            ['x1', 'x2', 'x3','x4', 'x5', 'x6','x7', 'x8', 'x9'])
plt.xlabel('Features')
plt.ylabel('Features value')
plt.show()

#Linear Regression model2
#split the data into trainning and testing set
X = data_for_model.iloc[:,0:-1]
y = data_for_model.iloc[:,-1]
X = X.values
y = y.values
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

#Second linear regression model, with 8 features
slr = LinearRegression()
slr.fit(X_train, y_train)

y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

#plot residual errors of LR model2
plt.scatter(y_train_pred,y_train_pred - y_train, 
            c = 'steelblue',marker = 'o',
            edgecolor = 'white',label = 'Trainning data')

plt.scatter(y_test_pred,y_test_pred - y_test, 
            c = 'limegreen',marker = 's',
            edgecolor = 'white',label = 'Test data')

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('residual errors')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = y.min(), xmax = y.max(), color = 'black',lw = 2)
plt.xlim = ([y.min(),y.max()])
plt.show()

#print performance indicators
print("MSE train: ")
print(round(mean_squared_error(y_train,y_train_pred),3))
print("MSE test: ")
print(round(mean_squared_error(y_test,y_test_pred),3))

print("R^2 train: ")
print(round(r2_score(y_train,y_train_pred),3))
print("R^2 test: ")
print(round(r2_score(y_test,y_test_pred),3))

#print model coefficients and intercept
print('coefficients:')
print(slr.coef_.tolist())
print('intercept:')
print(slr.intercept_.tolist())

#Ridge Regression 
X = data.iloc[:,0:-1]
y = data.iloc[:,-1]
X = X.values
y = y.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#set alpha range and step
alpha_min = 0.1;
alpha_max = 2.0;
alpha_step = 0.1
alphas = np.arange(alpha_min, alpha_max + alpha_step, alpha_step)

MSE_train = []
MSE_test = []
R_square_train = []
R_square_test = []

for i in range(len(alphas)):
    alpha_test = alphas[i]
    ridge = Ridge(alpha = alpha_test)
    ridge.fit(X_train, y_train)
    
    y_train_pred = ridge.predict(X_train)
    y_test_pred = ridge.predict(X_test)

    MSE_train.append(mean_squared_error(y_train,y_train_pred))
    MSE_test.append(mean_squared_error(y_test,y_test_pred))
    R_square_train.append(r2_score(y_train,y_train_pred))
    R_square_test.append(r2_score(y_test,y_test_pred))
    
    
#get performance indicators of different alpha
ridge_performance = pd.DataFrame({'alpha':alphas,'MSE_train':MSE_train,'MSE_test':MSE_test
                           ,'R_square_train':R_square_train,'R_square_test':R_square_test})

#build ridge model with best alpha
ridge = Ridge(alpha = 0.1)
ridge.fit(X_train, y_train)

y_train_pred = ridge.predict(X_train)
y_test_pred = ridge.predict(X_test)

#plot residual errors of ridge model
plt.scatter(y_train_pred,y_train_pred - y_train, 
            c = 'steelblue',marker = 'o',edgecolor = 'white',label = 'Trainning data')

plt.scatter(y_test_pred,y_test_pred - y_test, 
            c = 'limegreen',marker = 's',edgecolor = 'white',
            label = 'Test data')

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('residual errors')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = y.min(), xmax = y.max(), color = 'black',lw = 2)
plt.xlim = ([y.min(),y.max()])
plt.show()

#plot performance indicators
print("MSE train: ")
print(round(mean_squared_error(y_train,y_train_pred),3))
print("MSE test: ")
print(round(mean_squared_error(y_test,y_test_pred),3))

print("R^2 train: ")
print(round(r2_score(y_train,y_train_pred),3))
print("R^2 test: ")
print(round(r2_score(y_test,y_test_pred),3))

#print coefficients and intercept
print('coefficients:')
print(ridge.coef_.tolist())
print('intercept:')
print(ridge.intercept_.tolist())

#Lasso Regression 
X = data.iloc[:,0:-1]
y = data.iloc[:,-1]
X = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)
#set alpha range and step
alpha_min = 0.1;
alpha_max = 2.0;
alpha_step = 0.1
alphas = np.arange(alpha_min, alpha_max + alpha_step, alpha_step)

MSE_train = []
MSE_test = []
R_square_train = []
R_square_test = []

for i in range(len(alphas)):
    alpha_test = alphas[i]
    lasso = Lasso(alpha = alpha_test)
    lasso.fit(X_train, y_train)
    
    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)

    MSE_train.append(mean_squared_error(y_train,y_train_pred))
    MSE_test.append(mean_squared_error(y_test,y_test_pred))
    R_square_train.append(r2_score(y_train,y_train_pred))
    R_square_test.append(r2_score(y_test,y_test_pred))

#get performance indicators of different alpha
lasso_performance = pd.DataFrame({'alpha':alphas,'MSE_train':MSE_train,'MSE_test':MSE_test
                           ,'R_square_train':R_square_train,'R_square_test':R_square_test})

    
#build ridge model with best alpha
lasso = Lasso(alpha = 0.1)
lasso.fit(X_train, y_train)

y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)

#plot residual errors of lasso model
plt.scatter(y_train_pred,y_train_pred - y_train, 
            c = 'steelblue',marker = 'o',
            edgecolor = 'white',
            label = 'Trainning data')

plt.scatter(y_test_pred,y_test_pred - y_test, 
            c = 'limegreen',marker = 's',
            edgecolor = 'white',
            label = 'Test data')

#plot residual errors of ridge model
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('residual errors')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = y.min(), xmax = y.max(), color = 'black',lw = 2)
plt.xlim = ([y.min(),y.max()])
plt.show()

#plot performance indicators
print("MSE train: ")
print(round(mean_squared_error(y_train,y_train_pred),3))
print("MSE test: ")
print(round(mean_squared_error(y_test,y_test_pred),3))

print("R^2 train: ")
print(round(r2_score(y_train,y_train_pred),3))
print("R^2 test: ")
print(round(r2_score(y_test,y_test_pred),3))

#print coefficients and intercept
print('coefficients:')
print(lasso.coef_.tolist())
print('intercept:')
print(lasso.intercept_.tolist())


#ElasticNet Regression
X = data.iloc[:,0:-1]
y = data.iloc[:,-1]
X = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

#use GridSearch to find the best parames
tuned_parameters = [{'alpha':np.arange(0.01,1.1,0.1).tolist(),
                     'l1_ratio':np.arange(0.1,1.0,0.1).tolist()}]
clf=GridSearchCV(ElasticNet(max_iter = 1000),tuned_parameters,scoring='r2',cv=5) 

print('begin searching for best params:')
print()
clf.fit(X_train,y_train)
print('best params found:')
print(clf.best_estimator_)
print()


#build ridge model with best params
elanet = ElasticNet(alpha = 0.01, l1_ratio = 0.9)
elanet.fit(X_train, y_train)

y_train_pred = elanet.predict(X_train)
y_test_pred = elanet.predict(X_test)


#plot residual errors of ridge model
plt.scatter(y_train_pred,y_train_pred - y_train, 
            c = 'steelblue',marker = 'o',
            edgecolor = 'white',
            label = 'Trainning data')

plt.scatter(y_test_pred,y_test_pred - y_test, 
            c = 'limegreen',marker = 's',
            edgecolor = 'white',
            label = 'Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('residual errors')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = y.min(), xmax = y.max(), color = 'black',lw = 2)
plt.xlim = ([y.min(),y.max()])
plt.show()

#plot performance indicators
print("MSE train: ")
print(round(mean_squared_error(y_train,y_train_pred),3))
print("MSE test: ")
print(round(mean_squared_error(y_test,y_test_pred),3))

print("R^2 train: ")
print(round(r2_score(y_train,y_train_pred),3))
print("R^2 test: ")
print(round(r2_score(y_test,y_test_pred),3))


#print coefficients and intercept
print('coefficients:')
print(elanet.coef_.tolist())
print('intercept:')
print(elanet.intercept_.tolist())