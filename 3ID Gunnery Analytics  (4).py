#!/usr/bin/env python
# coding: utf-8

# # Raw Data Organization

# Load Packages 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.impute import SimpleImputer


# Disable Warnings 

# In[2]:


import warnings
warnings.filterwarnings("ignore")


# Read in initial raw data 

# In[3]:


df = pd.read_excel('Gunnery Data V1.xlsx')

df.head()


# Drop Crews that do not have Table VI Score 

# In[4]:


df['GTVI Score'].dropna(axis=0,inplace = True)


# Rename some poorly named columns 

# In[5]:


df = df.rename(columns = {'Crew ' : 'Crew','qual_M4M16556' : 'M4','qual_M99mmBerettaPistol' : 'M99',
                          'qual_XM17' : 'M17'  })


# Make Master Gunner Variable from string of ASIs 

# In[6]:


df['master_gunner'] = 0
for i in range(len(df['all_duty_additional_skills'])):
    if 'K8' in str(df['all_duty_additional_skills'][i]) or 'J3' in str(df['all_duty_additional_skills'][i]) or 'R8' in str(df['all_duty_additional_skills'][i]):
        df['master_gunner'][i] = 1


# Make Variables for other ASIs 

# In[7]:


df['maintanance_operations'] = 0
for i in range(len(df['all_duty_additional_skills'])):
    if 'K4' in str(df['all_duty_additional_skills'][i]) or 'B9' in str(df['all_duty_additional_skills'][i]):
        df['maintanance_operations'][i] = 1


# Make age variable out of age brackets

# In[8]:


df['age1'] = df['age_bracket'].str[0:2]
df['age2'] = df['age_bracket'].str[3:5]
df['age1'] = pd.to_numeric(df['age1'])
df['age2'] = pd.to_numeric(df['age2'])
df['avg_age'] = (df['age1'] + df['age2']) / 2

df.drop('age1', axis = 1, inplace = True)
df.drop('age2', axis = 1, inplace = True)


# Make a skill variable (1-3 for E and 4 for O)

# In[9]:


df['skill_level'] = df['individual_skill_level'].str[12]
df['O_E'] = df['grade_rank_code'].str[0]
df['O_E'] = df['O_E'].fillna('M')
for i in range(len(df['O_E'])):
    if df['O_E'] [i] == 'O':
        df['skill_level'][i] = '4'
df['skill_level'] = pd.to_numeric(df['skill_level'])


# Choose variables that are not nonsensical

# In[10]:


cols = ['Crew', 'Battalion', 'Vehicle', 'GTVI Score', 'GT VI Eng', 'Q1',
       'Position', 'VACANT', 'Sim Total Hrs','GTLF Score', 'GTLF Attmpt',
       'master_gunner', 'maintanance_operations', 'avg_age', 'apft_pass', 
       'apft_score_total','apft_score_su','apft_score_pu','apft_score_run','Sim Total Hrs','acft_score_total',
       'body_comp_pass', 'ethnic_group', 'height', 'skill_level',
       'O_E', 'M4', 'M99', 'M17', 'time_in_service_months',
       'weight']

df = df[cols]


# Make one hot variable for qual scores 

# In[11]:


#quals = pd.get_dummies(df[['M4','M99','M17']])
#print(quals.columns)
#df.reset_index()
#df = df.append(quals)


# Convert variable types 

# In[12]:


#for col in ['VACANT','GTVI Score','GT VI Eng','Q1','master_gunner','maintanance_operations','avg_age','apft_pass','apft_score_total','apft_score_su','apft_score_pu','apft_score_run','body_comp_pass','height', 'skill_level','Sim Total Hrs','acft_score_total','M4_EXPERT', 'M4_MARKSMAN', 'M4_SHARPSHOOTER', 'M4_UNQUALIFIED','asvab_gt_score','M99_EXPERT', 'M99_MARKSMAN', 'M99_SHARPSHOOTER', 'M99_UNQUALIFIED','M17_EXPERT', 'M17_MARKSMAN', 'M17_SHARPSHOOTER', 'M17_UNQUALIFIED','time_in_service_months','weight']:
#    if df[col].dtypes == 'object':
#        df[col] = df[col].astype('float')


# Drop empty columns 

# In[13]:


df = df.dropna(how = 'all', axis = 1)

df.columns


# # Data Widening (Make Crews)

# Make a crew df with crew wide variables (Again, sim data left out but should be included when data is available)

# In[14]:


crew_cols = ['Crew','Battalion','Vehicle',
        'GTVI Score','GT VI Eng','Q1']

crew_df = pd.DataFrame({'members' : df.groupby(crew_cols).size()}).reset_index()
crew_df.head()


# Pull data into each member type 

# In[15]:


member_df = df.copy()
driver_df = member_df.loc[member_df['Position'] == 'Driver']
loader_df = member_df.loc[member_df['Position'] == 'Loader']
gunner_df = member_df.loc[member_df['Position'] == 'Gunner']
commander_df = member_df.loc[member_df['Position'] == 'Commander']


# Combine crew and member data into one dataframe 

# In[16]:


dict = {'dr' : driver_df, 'ld' : loader_df, 'gn' : gunner_df, 'tc' : commander_df}
for key in dict.keys():
  df = dict[key]
  cols_new = []
  for col in df.columns:
    if col not in crew_cols:
      cols_new.append(key + '_' + col)
    else:
      cols_new.append(col)
  df.columns = cols_new
  crew_df = pd.merge(crew_df, df, on = crew_cols, how = 'left')

crew_df.columns


# Drop Crews that have a vacant Gunner or Commander 

# In[17]:


crew_df['gn_VACANT'].dropna(axis=0,inplace = True)
crew_df['tc_VACANT'].dropna(axis=0,inplace = True)


# Split dataframe in Bradley and Tank and drop variables that do not apply to new dataframes then reset indicies 

# In[18]:


M1_df = crew_df.loc[crew_df['Vehicle'] =='Tank']
M2_df = crew_df.loc[crew_df['Vehicle'] =='Bradley']

M1_df = M1_df.dropna(how = 'all', axis = 1)
M2_df = M2_df.dropna(how = 'all', axis = 1)

crew_df = crew_df.reset_index()
M1_df = M1_df.reset_index()
M2_df = M2_df.reset_index()


# Choose only varibales that make sense 

# Determine missing data 

# In[19]:


# Set missing variable percentage tolerance (0-1)
tol = .5 

# Bradley
b_tol_missing = []
b_missing_any = []
tol_M2 = tol * len(M2_df['GTVI Score'])
missing_data_M2 = M2_df.isnull()
col = M2_df.columns

count_M2 = 0

for column in col:
    miss = missing_data_M2[column].tolist()
    trues = miss.count(True)
    if trues > tol_M2:
        ##print(column)
        ##print(missing_data_M2[column].value_counts())
        ##print('')
        count_M2 = count_M2 +1
        b_tol_missing.append(column)
        M2_df.drop(column, axis = 1, inplace = True)
    if ((trues > 0) & (trues <= tol_M2)):
        b_missing_any.append(column)
        
# Tank 
t_tol_missing = []
t_missing_any = []
tol_M1 = tol * len(M1_df['GTVI Score'])
missing_data_M1 = M1_df.isnull()
col = M1_df.columns

count_M1 = 0

for column in col:
    miss = missing_data_M1[column].tolist()
    trues = miss.count(True)
    if trues > tol_M1:
        ##print(column)
        ##print(missing_data_M1[column].value_counts())
        ##print('')
        count_M1 = count_M1 +1
        t_tol_missing.append(column)
        M1_df.drop(column, axis = 1, inplace = True)
    if ((trues > 0) & (trues <= tol_M1)):
        t_missing_any.append(column)
        
print('The following variables have more than', tol*100, 'percent of the the data missing for Bradleys and will be removed: \n',b_tol_missing,'\n')
print('The following variables have more than', tol*100, 'percent of the the data missing for Tanks: \n',t_tol_missing,'\n')

print('The following variables have more than 1 missing value for Bradleys and will have the missing values imputed: \n',b_missing_any,'\n')
print('The following variables have more than 1 missing value for Tanks and will have the missing values imputed: \n',t_missing_any,'\n')


# Remove Variables IDed above as having insufficient data 

# Create function to impute missing quant variables with median and cat varibles with "Missing"

# In[20]:


mymedian_imputer = SimpleImputer(strategy='median')

# Bradley 
cat_b = []
quant_b = []
for col in b_missing_any:
    if M2_df[col].dtypes == 'object':
        M2_df[col] =M2_df[[col]].fillna(value=-1)
        M2_df = M2_df.astype({col : "str"})
        M2_df[col == '-1', col] = 'Missing'
        cat_b.append(col)
    else:
        mymedian_imputer.fit_transform(M2_df[[col]])
        M2_df[col] = mymedian_imputer.fit_transform(M2_df[[col]])
        quant_b.append(col)

        
# Tank
cat_t = []
quant_t = []
for col in t_missing_any:
    if M1_df[col].dtypes == 'object':
        M1_df[col] =M1_df[[col]].fillna(value=-1)
        M1_df = M1_df.astype({col : "str"})
        M1_df[col == '-1', col] = 'Missing'
        cat_t.append(col)
    else:
        mymedian_imputer.fit_transform(M1_df[[col]])
        M1_df[col] = mymedian_imputer.fit_transform(M1_df[[col]])
        quant_t.append(col)
        


# Create master keys

# In[21]:


M1_df['Master Key'] = M1_df['Battalion'] + ' ' + M1_df['Crew']


# Read in Sim data 

# In[22]:


AGTS = pd.read_csv('AGTS_Clean.csv')


# Left Join AGTS Data to Tank Dataset 

# In[23]:


M1_df = M1_df.merge(AGTS, on='Master Key', how = 'left')


# Write out cleaned data sets 

# In[24]:


M1_df.to_csv('M1_clean.csv')
M2_df.to_csv('M2_clean.csv')
crew_df.to_csv('data_clean.csv')


# # Data Analysis 

# Load packages 

# In[25]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.preprocessing import StandardScaler  
from sklearn.preprocessing import PolynomialFeatures  
from sklearn.pipeline import Pipeline            
from sklearn.compose import ColumnTransformer     
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import scipy as sp
import statsmodels.api as sm
from sklearn.pipeline import make_pipeline
import random

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# Functions that will help us in this seciton 

# In[26]:


# compute the vif for all given features 
#gstatic.com Ashir Nair 

def compute_vif(considered_features,df):
    
    X = df[considered_features]
    # the calculation of variance inflation requires a constant
    X['intercept'] = 1
    
    # create dataframe to store vif values
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['Variable']!='intercept']
    return vif


# Read in Datasets (This is so the Data organization and Data Analysis can be run as two seperate notebooks)

# In[27]:


df = pd.read_csv('data_clean.csv')
df_t = pd.read_csv('M1_clean.csv')
df_b = pd.read_csv('M2_clean.csv')


# Drop index and unnamed columns 

# In[28]:


df_b.drop('index', axis = 1, inplace = True)
df_b.drop('Unnamed: 0', axis =1, inplace = True)

df_t.drop('index', axis = 1, inplace = True)
df_t.drop('Unnamed: 0', axis = 1, inplace = True)


# Preview Bradley Dataframe 

# In[29]:


df_b.columns
df_b['GTVI Score'].astype('float')


# In[30]:


df_b.describe()


# Preview Tank Dataframe 

# In[31]:


df_t.head()


# In[32]:


df_t.describe()


# Tank Filtered AbsoluteCorrelation Heatmap

# In[33]:


tol = .75

df_t_corr = df_t.corr().abs().abs()
df_t_corr.dropna(how = 'all', axis = 1, inplace = True)
df_t_corr.dropna(how = 'all', axis = 0, inplace = True)
df_t_corr = df_t_corr[(df_t_corr >= .15 ) & (df_t_corr !=1.000)]
plt.figure(figsize=(30,10))
mask = np.triu(np.ones_like(df_t_corr, dtype=bool))
sns.heatmap(df_t_corr, annot=True,mask = mask, cmap="Reds")
plt.title('Tank Data Set Correlation Heat Map')

df_t_corr.head()


# Bradley Correlation Heatmap

# In[34]:


df_b_corr = df_b.corr().abs()
df_b_corr.dropna(how = 'all', axis = 1, inplace = True)
df_b_corr = df_b_corr[(df_b_corr >= .25) & (df_b_corr !=1.000) & (df_b_corr !=0.99)]
plt.figure(figsize=(30,10))
mask = np.triu(np.ones_like(df_b_corr, dtype=bool))
sns.heatmap(df_b_corr, annot=True, mask = mask, cmap="Blues")
plt.title('Bradley Data Set Correlation Heat Map')


# Identify colinear variables by first finding all variables with p vlaues above tol then running Variance Inflation Factor Analysis to determine which variables should be addressed 

# In[35]:


#Bradley 
concerning_b = []
for i in df_b_corr.columns:
    for j in df_b_corr.columns:
        if df_b_corr[i][j]>tol:
            concerning_b.append(j)
print('Bradley Variables with above tolerance correlation \n',concerning_b,'\n\n')


VIF_b = compute_vif(concerning_b,df_b).sort_values('VIF', ascending=True)
print(VIF_b)

#Tank
concerning_t = []
for i in df_t_corr.columns:
    for j in df_t_corr.columns:
        if df_t_corr[i][j]>tol:
            concerning_t.append(j)
print('Tank variables with above tolerance correlation \n',concerning_t,'\n\n')

VIF_t = compute_vif(concerning_b,df_b).sort_values('VIF', ascending=True)
print(VIF_t)


# Select Bradley varibales most highly correlated with GTVI Score

# In[36]:


#Bradley
corr_GTVI_Score_b = df_b_corr['GTVI Score'].abs().sort_values(ascending = False)
print('Bradley P values for GTVI Score \n')
for i in range(len(corr_GTVI_Score_b.index)):
    if corr_GTVI_Score_b.array[i] >.05:
        print(corr_GTVI_Score_b.index[i])
        print(corr_GTVI_Score_b[i])

vars_b = []
corr_GTVI_Score_b = df_b_corr['GTVI Score'].abs().sort_values(ascending = False)
for i in range(len(corr_GTVI_Score_b.index)):
    if corr_GTVI_Score_b.array[i] >.05:
        vars_b.append(corr_GTVI_Score_b.index[i])
        
print('Bradley variables with p>.1 for GTVI Score: \n', vars_b,'\n')


# Select Tank varibales most highly correlated with GTVI Score

# In[ ]:


#Tank
corr_GTVI_Score_t = df_t_corr['GTVI Score'].abs().sort_values(ascending = False)
print('Tank P values for GTVI Score \n')
for i in range(len(corr_GTVI_Score_t.index)):
    if corr_GTVI_Score_t.array[i] >.1:
        print(corr_GTVI_Score_t.index[i])
        print(corr_GTVI_Score_t[i])
        

vars_t = []
corr_GTVI_Score_t = df_t_corr['GTVI Score'].abs().sort_values(ascending = False)
for i in range(len(corr_GTVI_Score_t.index)):
    if corr_GTVI_Score_t.array[i] >.1:
        vars_t.append(corr_GTVI_Score_t.index[i])
        
print('Tank variables with p>.1 for GTVI Score: \n', vars_t,'\n')


# Select Bradley varibales most highly correlated with Q1

# In[ ]:


#Bradley
corr_Q1_b = df_b_corr['Q1'].abs().sort_values(ascending = False)
print('Bradley P values for Q1 \n')
for i in range(len(corr_Q1_b.index)):
    if corr_Q1_b.array[i] >.1:
        print(corr_Q1_b.index[i])
        print(corr_Q1_b[i])

vars_b_Q = []
corr_Q1_b = df_b_corr['Q1'].abs().sort_values(ascending = False)
for i in range(len(corr_Q1_b.index)):
    if corr_Q1_b.array[i] >.1:
        vars_b_Q.append(corr_Q1_b.index[i])
        
print('Bradley variables with p>.1 for Q1: \n', vars_b_Q,'\n')


# Select Tank varibales most highly correlated with Q1

# In[ ]:


#Tank
corr_Q1_t = df_t_corr['Q1'].abs().sort_values(ascending = False)
print('Tank P values for Q1 \n\n')
for i in range(len(corr_Q1_t.index)):
    if corr_Q1_t.array[i] >.1:
        print(corr_Q1_t.index[i])
        print(corr_Q1_t[i])

vars_t_Q = []
corr_Q1_t = df_t_corr['Q1'].abs().sort_values(ascending = False)
for i in range(len(corr_Q1_t.index)):
    if corr_Q1_t.array[i] >.1:
        vars_t_Q.append(corr_Q1_t.index[i])
        
print('Tank variables with p>.1 for Q1: \n', vars_t,'\n\n')


# # Tank Model Development 

# Training and Test Splits 

# In[ ]:


T_X = df_t[['dr_time_in_service_months', 'gn_apft_score_su', 'AVG Time Of Eng']]
T_Y = df_t['GTVI Score']
           

T_X_Train, T_X_Test, T_Y_Train, T_Y_Test = train_test_split(T_X,T_Y, test_size = .25, random_state = 5)


# Build Ridge Regression Pipeline 

# In[ ]:


Tank_Ridge_Input = [('Scale',StandardScaler()),('Polynomial',PolynomialFeatures(degree = 2,include_bias = False)),('model', Ridge())]

Brad_Ridge_Pipe = Pipeline(Tank_Ridge_Input)
Brad_Ridge_Pipe.fit(B_X_Train,B_Y_Train)

Brad_Ridge_Pipe.predict(B_X_Test)

### GRID SEARCH ###
parameter = [{'model__alpha': np.arange(0,1000,10)}]
Brad_grid = GridSearchCV(estimator = Brad_Ridge_Pipe, param_grid = parameter, cv = 5, n_jobs = -1)
Brad_grid.fit(B_X_Train,B_Y_Train)
Best_Brad = Brad_grid.best_estimator_
Best_Brad.get_params()
Best_Brad.score(B_X_Test,B_Y_Test)


# # Bradley Model Development

# Training / Test Splits 

# In[ ]:


B_X = df_b[['dr_body_comp_pass', 'dr_time_in_service_months', 'gn_apft_score_su', 'gn_maintanance_operations', 'gn_skill_level', 'dr_skill_level', 'dr_height', 'gn_apft_score_total']]
B_Y = df_b['GTVI Score']

B_X_Train, B_X_Test, B_Y_Train, B_Y_Test = train_test_split(B_X,B_Y, test_size = .25, random_state = 5)


# Build pipeline for a polynomial regression

# In[ ]:


Brad_Lin_Input = [('Scale',StandardScaler()),('Polynomial',PolynomialFeatures(degree = 2,include_bias = False)),('model', LinearRegression())]

Brad_Lin_Pipe = Pipeline(Brad_Lin_Input)

Brad_Lin_Pipe.fit(B_X_Train,B_Y_Train)
Brad_Lin_Pipe.score(B_X_Test,B_Y_Test)


# Build Pipeline for Ridge Regression

# In[ ]:


Brad_Ridge_Input = [('Scale',StandardScaler()),('Polynomial',PolynomialFeatures(degree = 2,include_bias = False)),('model', Ridge())]

Brad_Ridge_Pipe = Pipeline(Brad_Ridge_Input)
Brad_Ridge_Pipe.fit(B_X_Train,B_Y_Train)

Brad_Ridge_Pipe.predict(B_X_Test)

### GRID SEARCH ###
parameter = [{'model__alpha': np.arange(0,1000,10)}]
Brad_grid = GridSearchCV(estimator = Brad_Ridge_Pipe, param_grid = parameter, cv = 5, n_jobs = -1)
Brad_grid.fit(B_X_Train,B_Y_Train)
Best_Brad = Brad_grid.best_estimator_
Best_Brad.get_params()
Best_Brad.score(B_X_Test,B_Y_Test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




