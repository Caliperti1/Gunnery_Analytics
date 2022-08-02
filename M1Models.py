###################################################
# Coder: 1LT Chris Aliperti and CDT Joshua Wong
# Title: M1 Models 
# Date: 07JUL2022
# Version: 1.1
###################################################
# M1Models.py is a collection of models that attempt to determine relationships between
# the variables and the crew's gunnery performance and eventually develop a predictive model
# to give Commanders insight into how a new crew is predicted to perform based on previous data.

# In Version 1.1 the models are only developed using M1 data due to the lack of sim data for the Bradley.
# These functions provide a framework for models but more data is needed to properly train them and 
# determine which model is the most appropriate for this problem.

# This file is more or less a scratch book for the team to experiemnt with various models without having 
# to rewrite code to make minor adjustments or wory about changing the model code when the data is updated.

#%% Import Libraries and functions 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#%% m1_pca 
# m1_pca is a principle component analysis using the sim data for tanks. Principle component analysis 
# reduces the explantaroy varaibles into 2 or 3 'priniple components.' This allows the higher dimension data
# to be visualized in 2 or 3 dimensions. If the data is seperable this will reveal some clear delineation between
# groups of data which can then be used to create a model.

#### features (line 41) can be manipulated to choose which variables are included in the analysis 

#   Inputs:
#       df - cleaned M1 dataframe

#   Outputs:
#       finalDf - dataframe containing the 3 principle components and the dependent variable  
#       explained_variance_ratio_ - EVR is a meaure of how much variance in the dependent variable is explained
#           by each of the principle components. 
#       components_ - shows how much of an impact each of the orignal variables has on each component. 
#           (This is essentially the eigen vectors for each principle component)
     
def m1_pca(df):

    features = ['Time','Tot Hits','AVG Time To ID','AVG Time Of Eng']
    target = 'GTVI Score'

    # separating out the features
    x = df.loc[:, features].values

    # separating out the target
    y = df.loc[:,[target]].values

    # Standardized data (mean of 0, variance of 1)
    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    # PCA Projection to 2D
    pca = PCA(n_components = 3)
    principalComponents = pca.fit_transform(x)

    # Creates Data Frame of the 2 Principal Components
    principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2','PC3'])

    # Concatinate Principal Component data frame with target data frame
    finalDf = pd.concat([principalDf, df[['GTVI Score']]], axis=1)

    return finalDf, pca.explained_variance_ratio_, pca.components_

#%% m1_pc_selection
# Function that uses one of the many methods of determining the optimal number of principle 
# components to be included in a model. Creates an elbow plot that will allow the modeler to make 
# a decision about how many principle components to retain. This funciton should be used before the 
# previous function in order to determine how many PCs (line 66)

#   Inputs:
#       df - clean M1 dataframe

#   Outputs:
#       perc_var - percent variation explained by each additional principle component added.
#       figure - elbow plot that shows 'elbow' at the optimal primciple component number 
def m1_pc_selection(df):

    features = ['Time','Tot Hits','AVG Time To ID','AVG Time Of Eng']
    target = 'GTVI Score'

    x = df.loc[:, features].values

    y = df.loc[:,[target]].values

    x = StandardScaler().fit_transform(x)

    #scale predictor variables
    pca = PCA()
    X_reduced = pca.fit_transform(scale(x))

    #define cross validation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    regr = LinearRegression()
    mse = []

    # Calculate MSE with only the intercept
    score = -1*model_selection.cross_val_score(regr,
            np.ones((len(X_reduced),1)), y, cv=cv,
            scoring='neg_mean_squared_error').mean()    
    mse.append(score)

    # Calculate MSE using cross-validation, adding one component at a time
    for i in np.arange(1, 6):
        score = -1*model_selection.cross_val_score(regr,
                X_reduced[:,:i], y, cv=cv, scoring='neg_mean_squared_error').mean()
        mse.append(score)
        
    # Plot cross-validation results    
    plt.plot(mse)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('MSE')
    plt.title('GTVI Score')

    # The percent variance explained by adding each PC to model
    perc_var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    
    return perc_var

#%% m1_pcr_model
# Creates a principle component regression model. A PCR is similar to a linear regression
# where x_i is a principle component.

#   Inputs: 
#       df - clean M1 dataframe 

#   Outputs:
#       regr - principle component regression oject 
#       rmse - root mean square error of model 

def m1_pcr_model(df):

    features = ['Time','Tot Hits','AVG Time To ID','AVG Time Of Eng']
    target = 'GTVI Score'

    x = df.loc[:, features].values

    y = df.loc[:,[target]].values

    # split the dataset into training (70%) and testing (30%) sets
    X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0) 

    #scale the training and testing data
    pca = PCA()
    X_reduced_train = pca.fit_transform(scale(X_train))
    X_reduced_test = pca.transform(scale(X_test))[:,:1]

    #train PCR model on training data 
    regr = LinearRegression()
    regr.fit(X_reduced_train[:,:1], y_train)

    #calculate RMSE
    pred = regr.predict(X_reduced_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    return regr, rmse

#%% m1_reg_model
# Simple linear regression model (with current features using only sim data).

#   Inputs:
#       df - clean M1 dataframe 

#   Outputs:
#       regr - linear regression object 
#       rmse - root mean squre error of model


def m1_reg_model(df):

    features = ['Time','Tot Hits','AVG Time To ID','AVG Time Of Eng']
    target = 'GTVI Score'

    x = df.loc[:, features].values

    y = df.loc[:,[target]].values

    # split the dataset into training (70%) and testing (30%) sets
    X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0) 

    regr = LinearRegression()
    regr.fit(X_train, y_train)

    #calculate RMSE
    pred = regr.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    return regr, rmse

#%% Multi_reg
# Mutliple regression that allows features and targets to be maniplualted in finction input 

#   Inputs:
#       df - Clean M1 dataframe 
#       target - string containing target variable name 
#       features - list of strings containing feature names 

# Outpus:
#       regr - multiple regression object 
#       regr.coef_  - list of coefficeints for all features 

def multi_reg(df,target,features):

    x = df[features]
    y = df[target]

    regr = LinearRegression()
    regr.fit(x, y)

    return regr, regr.coef_

#%% m1_reg_predict 
# predicts scores using a regression model object (generated in previous functions)

#   Inputs:
#       regr - regression object 
#       predVals - default predicted values (inital guess)

#   Outputs:
#       predictedScore - predicted score 

def m1_reg_predict(regr, predVals):

    predictedScore = regr.predict([predVals])

    return predictedScore

#%% m1_pcr_predict 
# predicts scores using a principle component regression model object (generated in previous functions)

#   Inputs:
#       regr - regression object 
#       predVals - default predicted values (inital guess)

#   Outputs:
#       predictedScore - predicted score 

def m1_pcr_predict(regr, predVals):

    x = [predVals]

    pca = PCA()
    X_reduced = pca.fit_transform(scale(x))[:,:1]

    pred = regr.predict(X_reduced)

    return pred

#%% main
# runs models currently generated in M1 models and provides scores to allow analyst to choose best model to move forward with.

#   Inputs:
#       NONE

#   Outputs:
#       elbow plot to choose number of principle components 
#       regression RMSE
#       PCR regression RMSE
#       predicted score by simple regression
#       predicted score by PC regression
#       princple components 

def main():

    # Default Predicted Values
    predVals = [500.0, 300.0, 8.25, 9.34]

    # Read In filtered tank csv file
    df = pd.read_csv('M1_clean.csv')

    # Plot to choose pricipal components for pcr
    m1_pc_selection(df)
    plt.show()

    # output regression model and root mean squared error for model
    regr1, rmse1 = m1_reg_model(df)
    regr2, rmse2 = m1_pcr_model(df)


    print(f'Regression RMSE: {rmse1}')
    print(f'PCR RMSE: {rmse2}')

    predVals = []
    inputs = ['Time','Tot Hits','AVG Time To ID','AVG Time Of Eng']

    # User Input for Predicted Score Based of Values
    #for i in range(4):
    #    userInput = float(input(f'Input {inputs[i]} as a float: '))
    #    predVals.append(userInput)

    # Print out predicted score for each model based on predicted values
    print(f'Linear Regression Predicted Score: {m1_reg_predict(regr1,predVals)}')
    print(f'PCR Predicted Score: {m1_pcr_predict(regr2,predVals)}')

    
    a, b, c = m1_pca(df)

    print(a.head())
    print(b)
    print(c)




