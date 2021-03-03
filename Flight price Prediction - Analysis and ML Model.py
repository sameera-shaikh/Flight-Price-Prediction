#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir('C:/Users/sameera/OneDrive/Desktop/Learning/Analysis/Accelya/bsp-anon')
from sklearn import preprocessing
import re
from scipy.stats import zscore
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn import utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression,ElasticNet,Lasso,Ridge,BayesianRidge
from lightgbm.sklearn import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,AdaBoostRegressor,BaggingRegressor,VotingClassifier
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn import neighbors
from sklearn import svm

from sklearn.model_selection import KFold,train_test_split
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Read clean file
df1 = pd.read_csv('Clean_File.csv')
df1.columns


# In[4]:


pip install sweetviz


# In[4]:


import sweetviz as sv

my_report = sv.analyze(df1)
my_report.show_html()


# In[4]:


pd.set_option('display.max_columns', 100)
df1.head()


# In[5]:


df1['journey_strng'] = df1['bki63_orac'] + '-' + df1['bki63_dstc']

df1.head()


# In[6]:


#plot to check transaction currency countwise 

import matplotlib.pyplot as plt
import seaborn as sns

df1['curcode'].value_counts().plot(kind='bar', figsize = (8,3))


# In[7]:


ax = sns.stripplot(y = 'curcode', x = 'Amount(USD)', data = df1, size = 5, linewidth = 1)


# In[8]:


#stripplot to check distribution of cuurency


plt.figure(figsize=(8, 8))
ax3 = sns.stripplot(y='curcode', x='Amount(USD)', size = 4, data=df1, linewidth = 0.5)
plt.show


# In[9]:


#checking outlier

plt.figure(figsize=(16, 6))
sns.boxplot(data = df1, x = 'curcode', y = 'Amount(USD)')


# In[10]:


#flight departure time wise 

df1['departure_type'].value_counts().sort_values(ascending = True).plot(kind='barh', figsize = (8,3))


# In[10]:


#top 10 busiest route

plt.figure(figsize=(5, 5))
df1['journey_strng'].value_counts()[:10].sort_values(ascending = True).plot(kind='barh', figsize = (16,7))


# In[11]:


df2 = df1[['bks24_dais', 'Amount(USD)']]
df2['bks24_dais']=pd.to_datetime(df2['bks24_dais'])

df2['bks24_dais'] = pd.to_datetime(df2['bks24_dais']).dt.to_period('M')

table = pd.pivot_table(df2, values='Amount(USD)', index=['bks24_dais'],aggfunc=np.sum)

df2.set_index('bks24_dais',inplace=True)


# In[12]:


#the timeseries graph shows that the sale is not stattionary  

plt.figure(figsize = (16,16))
table.plot()


# In[13]:


#Flight price prediction for flights having one segment
#Filtering dataset
df = df1[df1['n_segments'] == 1]


# In[14]:


#checking general descriptive statistics for numerical variables
df.describe()


# In[15]:


#Create histogram to check the distribution. Its positively skewed data, whicn needs to be converted on logscale.

plt.figure(figsize=(8,5))
plt.hist(df1['Amount(USD)'],bins=10,color='b')
plt.title('Histogram of Flight Price')
plt.show()


# In[21]:


#converting amount to log scale
import numpy as np

df1['Amount(USD)'] = np.log(df1['Amount(USD)'])


# In[22]:


#Now the data is not skewed

plt.figure(figsize=(8,8))
plt.hist(df1['Amount(USD)'],bins=10,color='b')
plt.title('Histogram of Flight Price')
plt.show()


# In[23]:


#checking outliers


plt.figure(figsize=(16, 6))
sns.boxplot(data = df, x = 'curcode', y = 'Amount(USD)')


# In[24]:


#dropping outlier
q_low = df["Amount(USD)"].quantile(0.03)
q_hi  = df["Amount(USD)"].quantile(0.97)

df_filtered = df[(df["Amount(USD)"] < q_hi) & (df["Amount(USD)"] > q_low)]


# In[25]:


#dataframe shape before and after dropping outlier

print(df_filtered.shape)
print(df.shape)


# In[29]:


df_filtered.head()


# In[26]:


df_filtered.corr()

#no strong strong correlartion exists


# In[27]:


#no strong strong correlartion exists - snspairplot 

pairplot = df_filtered[['booking_travel_days_diff', 'Amount(USD)', 'label_amount', 'departure_hour']]
sns.pairplot(pairplot)


# In[25]:


df_filtered.columns


# In[26]:


corr = pairplot.corr()
corr


# In[30]:


#extractinf year, day of year from booking and departure date

df_filtered['doj_year']=pd.to_datetime(df_filtered['bki63_ftda']).dt.year
df_filtered['booking_year']=pd.to_datetime(df_filtered['bks24_dais']).dt.year
df_filtered['doj_dayofyear']=pd.to_datetime(df_filtered['bki63_ftda']).dt.dayofyear
df_filtered['booking_dayofyear']=pd.to_datetime(df_filtered['bks24_dais']).dt.dayofyear
df_filtered['year_month_DOJ']=df_filtered['doj_year'].astype(str).str.cat(df_filtered['doj_dayofyear'].astype(str),sep='.').astype(float)
df_filtered['year_month_Booking']=df_filtered['booking_year'].astype(str).str.cat(df_filtered['booking_dayofyear'].astype(str),sep='.').astype(float)


df_model = df_filtered[['airlinecode', 'bks24_todc', 'bki63_cabi', 'bki63_rbkd', 'bks39_rdii', 'bki63_segi', 'bki63_stpo', 'bks24_rfic',
                'n_tktt', 'label', 'label_amount', 'Amount(USD)', 'booking_travel_days_diff', 'journey_strng', 'departure_hour',
                'year_month_DOJ', 'year_month_Booking']]


# In[38]:


df_model.fillna('Not Available', inplace = True)


# In[50]:


df_model.head()


# In[48]:


df_model['bki63_cabi'].replace('Not Available', 0, inplace = True)


# In[49]:


#convert categorical variable to numberical using label encoder

le = preprocessing.LabelEncoder()

df['airlinecode_en'] = le.fit_transform(df['airlinecode'])

df_model['airlinecode_en']=le.fit_transform(df_model['airlinecode'])
df_model['bks24_todc_en']=le.fit_transform(df_model['bks24_todc'])
df_model['bki63_cabi_en']=le.fit_transform(df_model['bki63_cabi'])
df_model['bki63_rbkd_en']=le.fit_transform(df_model['bki63_rbkd'])
df_model['bks39_rdii_en']=le.fit_transform(df_model['bks39_rdii'])
df_model['bki63_stpo_en']=le.fit_transform(df_model['bki63_stpo'])
df_model['bks24_rfic_en']=le.fit_transform(df_model['bks24_rfic'])
df_model['label_en']=le.fit_transform(df_model['label'])
df_model['journey_strng_en']=le.fit_transform(df_model['journey_strng'])

model_data = df_model[['airlinecode_en', 'bks24_todc_en',
                       'bki63_cabi_en', 'bki63_rbkd_en', 'bks39_rdii_en', 
                       'bki63_stpo_en', 'bki63_segi', 'n_tktt', 'label_amount', 
                       'Amount(USD)', 'booking_travel_days_diff', 'bks24_rfic_en', 'label_en', 'journey_strng_en',
                       'departure_hour', 'year_month_DOJ', 'year_month_Booking']]


# In[51]:


#defining x and y variables

x_var = ['airlinecode_en', 'bks24_todc_en',
    'bki63_cabi_en', 'bki63_rbkd_en', 'bks39_rdii_en', 
    'bki63_stpo_en', 'bki63_segi', 'n_tktt', 'label_amount', 
    'booking_travel_days_diff', 'bks24_rfic_en', 'label_en', 'journey_strng_en',
    'departure_hour', 'year_month_DOJ', 'year_month_Booking']

X = model_data[x_var]
y = model_data['Amount(USD)']
    


# In[52]:


#test and train split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, random_state=1)

X_train.shape


# In[54]:


model_data.info()


# In[53]:


#fetaure standardization using standard scaler

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform (X_test)

# X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
# X_test_scaled = pd.DataFrame(X_test_scaled, index = X_test.index, columns = X_test.columns)

# train = pd.concat([X_train_scaled, y_train], axis=1)
# test = pd.concat([X_test_scaled, y_test], axis=1)


# In[47]:


X_train_scaled.head()


# In[37]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train_scaled, y_train)

y_pred = regressor.predict(X_test_scaled)


# In[43]:


print(regressor.coef_)


# In[44]:


print(regressor.intercept_)


# In[46]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred1 = regressor.predict(X_test)


r2_score(y_test, y_pred1)


# In[48]:


model_data.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[45]:


from sklearn.metrics import r2_score

r2_score(y_test, y_pred)


# In[34]:


X_train_scaled.head()


# In[ ]:


train.shape


# In[ ]:


#defining function to calculate RMSE 

def RMSE(estimator,X_train, Y_train, cv,n_jobs=-1):
    cv_results = cross_val_score(estimator,X_train,Y_train,cv=cv,scoring="neg_mean_squared_error",n_jobs=n_jobs)
    return (np.sqrt(-cv_results)).mean()


# In[70]:


#creating model

def baseModels(train_X,train_y):
    model_EN=ElasticNet(random_state=0)
    model_SVR=svm.SVR(kernel='rbf',C=0.005)
    model_Lasso=Lasso(alpha=0.1,max_iter=1000)
    model_Ridge=Ridge(alpha=0.1)
    model_Linear=LinearRegression()
    model_XGB = xgb.XGBRegressor(n_estimators=100, learning_rate=0.02, gamma=0, subsample=0.75,colsample_bytree=1, max_depth=4)
    model_DTR = DecisionTreeRegressor(max_depth=4,min_samples_split=5,max_leaf_nodes=10)
    model_RFR=RandomForestRegressor(n_jobs=-1)
    model_KNN=neighbors.KNeighborsRegressor(3,weights='uniform')
    model_Bayesian=BayesianRidge()
    model_adaboost=AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0, loss='linear', random_state=None)
    kf = KFold(n_splits=5, random_state=None, shuffle=True)

    models={'ElasticNet':model_EN,'SVR':model_SVR,'Lasso':model_Lasso,'Ridge':model_Ridge,
            'XGB':model_XGB,'DTR':model_DTR,'RandomForest':model_RFR,'KNN':model_KNN,
            'Bayes':model_Bayesian,'Linear':model_Linear,'AdaBoost':model_adaboost}

    rmse=[]
    for model in models.values():

        rmse.append(RMSE(model,train_X,train_y,kf))                         
    dataz = pd.DataFrame(data={'RMSE':rmse},index=models.keys())
    return  dataz


# In[71]:


#Random Forest gives least RMSE, so will proceed with Random Forest hyperparamter tuning

baseModels(X_train,y_train)


# In[48]:


model_rf=RandomForestRegressor(n_jobs=-1)
RMSE(model_rf,X,y,10)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 42)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())


# In[ ]:


# # Use the random grid to search for best hyperparameters
# # First create the base model to tune
# rf = RandomForestRegressor()
# # Random search of parameters, using 3 fold cross validation, 
# # search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# # Fit the random search model
# rf_random.fit(X_train_scaled, y_train)


# In[50]:


from sklearn.ensemble import RandomForestRegressor

random_forest = RandomForestRegressor()
random_forest.fit(X_train_scaled, y_train)
print(random_forest.score(X_train_scaled, y_train))

from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(random_forest.get_params())


# In[51]:


#Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 20, stop = 200, num = 5)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(1, 45, num = 3)]
# Minimum number of samples required to split a node
min_samples_split = [5, 10]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split}

pprint(random_grid)


# In[52]:


from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint

forest = RandomForestRegressor(n_jobs=-1)

rf_random = RandomizedSearchCV(estimator = forest, param_distributions = random_grid, n_iter = 10, cv = 10, verbose=2, random_state=42, n_jobs = -1, scoring='neg_mean_squared_error')
# Fit the random search model
rf_random.fit(X_train_scaled, y_train)


# In[53]:


cvres2 = rf_random.cv_results_
for mean_score, params in zip(cvres2["mean_test_score"], cvres2["params"]):
    print(np.sqrt(-mean_score), params)


# In[54]:


rf_random.best_estimator_


# In[55]:


rf_random.best_params_


# In[56]:


rf_random.best_estimator_


# In[59]:



# extract the numerical values of feature importance from the grid search search
importances = rf_random.best_estimator_.feature_importances_

#create a feature list from the original dataset (list of columns)
# What are this numbers? Let's get back to the columns of the original dataset
feature_list = list(X.columns)

#create a list of tuples
feature_importance= sorted(zip(importances, feature_list), reverse=True)

#create two lists from the previous list of tuples
df = pd.DataFrame(feature_importance, columns=['importance', 'feature'])
importance= list(df['importance'])
feature= list(df['feature'])

print(df)


# In[60]:


plt.figure(figsize = (10,10))
# list of x locations for plotting
x_values = list(range(len(feature_importance)))

# Make a bar chart
plt.figure(figsize=(8,5))
plt.bar(x_values, importance, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');


# In[63]:


max_depths = np.linspace(1, 50, 50, endpoint=True)

train_results = []
test_results = []

for i in max_depths:
    dt = RandomForestRegressor(max_depth=i)
    dt.fit(X_train_scaled, y_train)    
    #compute accuracy for train data
    housing_tree = dt.predict(X_train_scaled)
    errors = abs(housing_tree - y_train)
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_train)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    #append results of accuracy
    train_results.append(accuracy)
    
     #now again for test data
    housing_tree = dt.predict(X_test_scaled)
    errors = abs(housing_tree - y_test)
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_test)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    #append results of accuracy
    test_results.append(accuracy)
    
    
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, 'b', label='Train accuracy')
line2, = plt.plot(max_depths, test_results, 'r', label= 'Test accuracy')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Accuracy score')
plt.xlabel('Tree depth')


##The Test- train accuracy graph splits at 9, so will have 8 features in the model, based on importance

#The train graph becomes flat at tree depth of ~21. So, will have max_depth as 21.


# In[57]:


random_best= rf_random.best_estimator_.predict(X_train_scaled)
errors = abs(random_best - y_train)
# Calculate mean absolute percentage error (MAPE)
mape = np.mean(100 * (errors / y_train))
# Calculate and display accuracy
accuracy = 100 - mape    
#print result
print('The best model from the randomized search has an accuracy of', round(accuracy, 2),'%')


# In[58]:


#this is the RMSE

from sklearn.metrics import mean_squared_error
final_mse = mean_squared_error(y_train, random_best)
final_rmse = np.sqrt(final_mse)
print('The best model from the randomized search has a RMSE of', round(final_rmse, 2))


# In[61]:


# Evaluate best model on the test set

final_model = rf_random.best_estimator_
# Predicting test set results
final_pred = final_model.predict(X_test_scaled)
final_mse = mean_squared_error(y_test, final_pred)
final_rmse = np.sqrt(final_mse)
print('The final RMSE on the test set is', round(final_rmse, 2))


# In[62]:


errors = abs(final_pred - y_test)
# Calculate mean absolute percentage error (MAPE)
mape = np.mean(100 * (errors / y_test))
# Calculate and display accuracy
accuracy = 100 - mape    
#print result
print('The best model achieves on the test set an accuracy of', round(accuracy, 2),'%')

