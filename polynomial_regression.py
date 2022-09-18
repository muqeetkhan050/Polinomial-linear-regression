# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 16:31:39 2022

@author: Muqeet
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import fuzzywuzzy
from fuzzywuzzy import process

data=pd.read_csv("Position_Salaries.csv")

data.head()
#%%
X=data.iloc[:,1:2].values

y=data.iloc[:,2].values


#%%

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#%%
##fitting linear regression in dataset 


from sklearn.linear_model import LinearRegression
linregressor=LinearRegression()
linregressor.fit(X,y)
A=linregressor.predict(X)

#%%

#making a variable X with polynomial means their square etc
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)



#%%

##fitting polynomial regression in dataset
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)
B=lin_reg2.predict(X_poly)

#%%

#visualising the linear regression result
plt.scatter(X,y,color="red")
plt.plot(X,A,color="blue")
plt.title("truth or bluff(Linear regressor plot")
plt.xlabel("Position level")
plt.ylabel("salary")
plt.show()

#%%

#visualising polynomial regression model
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1 ))
plt.scatter(X,y,color="red")
plt.plot(X,B,color="blue")
plt.title("truth or bluff(polynomial regression plot)")
plt.xlabel("position level")
plt.ylabel("salary")
plt.show()

#%%

#predicting a new result with linear regression 

linregressor.predict(6)