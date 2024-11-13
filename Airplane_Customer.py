#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install xgboost


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay, classification_report
warnings.filterwarnings('ignore')


# In[ ]:


airplane_customer = pd.read_csv(r"C:\Users\Rahul\OneDrive\Desktop\Airline Customer\Airline_customer_satisfaction.csv")


# In[ ]:


def set_size_style(width, height, style=None):
    plt.figure(figsize=(width, height))
    if style != None:
        sns.set_style(style)
        
        
def customize_plot(plot, title:str, xlabel:str,  ylabel:str, title_font:int, label_font:int):
    plot.set_title(title, fontsize = title_font, weight='bold')
    plot.set_xlabel(xlabel, fontsize = label_font, weight='bold')
    plot.set_ylabel(ylabel, fontsize = label_font, weight='bold')


# # DATA EXPLORATION AND CLEANING

# In[5]:


airplane_customer


# In[6]:


airplane_customer.sample(3)


# In[7]:


airplane_customer.head(3)


# In[8]:


airplane_customer.tail(3)


# In[9]:


airplane_customer.shape


# In[10]:


airplane_customer.info()


# In[11]:


airplane_customer.describe()


# In[12]:


airplane_customer.describe(include = 'object')


# In[13]:


for col in airplane_customer.describe(include='object').columns:
    print('Column Name: ',col)
    print(airplane_customer[col].unique())
    print('-'*50)


# # HANDLING NULL VALUES

# In[14]:


airplane_customer.isna().sum()


# In[15]:


airplane_customer['Arrival Delay in Minutes'].fillna(airplane_customer['Arrival Delay in Minutes'].mean(), inplace=True)


# # HANDLING OUTLIERS

# In[16]:


for col in airplane_customer.describe().columns:
    set_size_style(16,2,'ticks')
    sns.boxplot(data=airplane_customer, x=col)
    plt.show()


# In[17]:


airplane_customer = airplane_customer.drop(airplane_customer[airplane_customer['Departure Delay in Minutes'] > 500 ].index)
airplane_customer = airplane_customer.drop(airplane_customer[airplane_customer['Arrival Delay in Minutes'] > 500 ].index)
airplane_customer = airplane_customer.drop(airplane_customer[airplane_customer['Flight Distance'] > 5500 ].index)
airplane_customer.reset_index(drop=True, inplace=True)
airplane_customer.shape


# # EXPLORATORY DATA ANALYSIS

# In[18]:


airplane_customer.columns


# In[19]:


set_size_style(10,5)
ax = sns.histplot(airplane_customer['Age'],bins=25,color= sns.color_palette('Spectral')[0],kde=True)
customize_plot(ax,'Age Distribution','Age','Frequency',13,10)


# The majority of individuals fall within the age range of 20 to 60 years, with a notable concentration around the age of 40.

# In[20]:


plt.title("Satisfied vs Dissatisfied", fontsize = 12, weight='bold')
plt.pie(airplane_customer['satisfaction'].value_counts(),
        labels=airplane_customer['satisfaction'].value_counts().index,
        radius=1, autopct='%.2f%%',textprops={'fontsize': 10, 'fontweight': 'bold'}, 
        colors = sns.color_palette('Spectral'))
plt.show()


#  The number of satisfied customers exceeds the number of dissatisfied customers, indicating a prevailing trend towards positive experiences with the service or product.

# In[21]:


set_size_style(12,5)
age_groups = airplane_customer.groupby('Age')['satisfaction'].value_counts(normalize=True).unstack()
satisfied_percentage = age_groups['satisfied'] * 100
ax =sns.lineplot(x=satisfied_percentage.index, y=satisfied_percentage.values, marker='o', color= sns.color_palette('Spectral')[0])
customize_plot(ax, 'Satisfied Percentage across Age', 'Age', 'Satisfied Percentage',13,10)
plt.grid(True)
plt.show()


# * Individuals in their 40s and 50s exhibit satisfaction with airline services.
# * Conversely, older individuals above the age of 70 express significantly higher levels of dissatisfaction with the services provided

# In[22]:


set_size_style(12,7)
class_ratings = airplane_customer.groupby('Class').agg({'Cleanliness':'mean',
                                                       'Checkin service' : 'mean',
                                                       'Seat comfort':'mean',
                                                       'Inflight wifi service':'mean', 
                                                       'Leg room service':'mean'}).reset_index()
class_ratings_melted = pd.melt(class_ratings, id_vars='Class', var_name='Category', value_name='Mean Rating')
ax = sns.barplot(x='Class', y='Mean Rating', hue='Category', data=class_ratings_melted, palette='Spectral')
for c in ax.containers:
        ax.bar_label(c)
customize_plot(ax, 'Mean Ratings across Class', 'Class', 'Mean Rating',13,10)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')


# - Travelers in the business class generally give higher average ratings for cleanliness, check-in experience, in-flight wifi, and legroom service.
# - Interestingly, passengers in the business class tend to rate seat comfort comparatively lower.

# In[23]:


corr = airplane_customer[['Flight Distance', 'Seat comfort', 'Departure/Arrival time convenient',
       'Food and drink', 'Gate location', 'Inflight wifi service',
       'Inflight entertainment', 'Online support', 'Ease of Online booking',
       'On-board service', 'Leg room service', 'Baggage handling',
       'Checkin service', 'Cleanliness', 'Online boarding',
       'Departure Delay in Minutes', 'Arrival Delay in Minutes']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(12, 10))
sns.heatmap(corr, mask=mask, center=0, linewidths=1, annot=True, fmt=".2f",cmap = 'coolwarm')


# In[24]:


airplane_customer.drop(columns = ['Arrival Delay in Minutes'],inplace = True)


# # ENCODING CATEGORICAL FEATURES

# In[25]:


dummies=pd.get_dummies(airplane_customer['Class'], dtype=int)
dummies


# In[26]:


cust_encoded = pd.concat([airplane_customer,dummies], axis = 'columns')
cust_encoded.drop(columns = ['Class'], inplace=True)
cust_encoded


# In[27]:


cust_encoded['Customer Type'] = cust_encoded['Customer Type'].map({'Loyal Customer': 1, 'disloyal Customer': 0})
cust_encoded['Type of Travel'] = cust_encoded['Type of Travel'].map({'Personal Travel': 1, 'Business travel': 0})
cust_encoded['satisfaction'] = cust_encoded['satisfaction'].map({'satisfied': 1, 'dissatisfied': 0})

cust_encoded


# # SPLITTING DATA

# In[28]:


X = cust_encoded.drop(columns = ['satisfaction'])
Y = cust_encoded['satisfaction']


# In[29]:


X.shape,Y.shape


# In[30]:


X


# In[31]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2, random_state=42)
X_train.shape, Y_train.shape


# # SCALING DATA

# In[32]:


scaler = StandardScaler()


# In[33]:


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# # SELECTING BEST MODEL 

# In[34]:


models = {
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Random Forest Classifier': RandomForestClassifier(),
    'Decision Tree Classifier': DecisionTreeClassifier(),
}
results = []

for name, model in models.items():
    kf = KFold(n_splits=6, shuffle=True, random_state=42)
    cv_results = cross_val_score(model, X_train_scaled, Y_train, cv=kf, scoring='accuracy')
    print(f'CV Score (Mean) {name}: {np.mean(cv_results)}')
    results.append(cv_results)

plt.boxplot(results, labels=models.keys())
plt.title('Cross-validation Scores for Classification Models')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.show()


# - Random Forest Classifier outperforms other classification models

# In[35]:


rf = RandomForestClassifier(random_state=42)


# In[36]:


rf.fit(X_train,Y_train)


# In[37]:


Y_pred = rf.predict(X_test)


# # MODEL EVALUATION

# In[38]:


plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(rf, X_test, Y_test, cmap='Reds', values_format='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[39]:


report = classification_report(Y_test, Y_pred)
print(report)


# - The Random Forest Model achieves high precision, recall, and F1-score for both classes, indicating that it performs well in classifying both dissatisfied and satisfied customers.
# - The overall accuracy of 96% suggests that the model is accurate in predicting the customer satisfaction status.
# - Let's try Extreme Gradient Boosting

# # EXTREME GRADIENT BOOSTING

# In[41]:


customer_dmatrix = xgb.DMatrix(data=X_train_scaled,label=Y_train)
params={'binary':'logistic'}
cv_results = xgb.cv(dtrain=customer_dmatrix,
                    params=params,
                    nfold=4,
                    metrics="error",
                    as_pandas=True,
                    seed=42)


# In[42]:


cv_results['test-accuracy-mean'] = 1 - cv_results['test-error-mean']
mean_accuracy = cv_results['test-accuracy-mean'].iloc[-1]
print("Mean Accuracy (CV):", mean_accuracy)


# - Now, let's proceed with hyperparameter tuning to improve the accuracy

# # HYPERPARAMETER TUNING

# ## GRID SEARCH CV

# In[43]:


xgb_param_grid = {
    'learning_rate': [0.01, 0.1, 0.5],
    'n_estimators': [200],
    'max'
    'subsample': [0.3, 0.5, 0.9]
}
xgboost_model = xgb.XGBClassifier(objective = 'binary:logistic',seed=42)
grid_xgboost = GridSearchCV(
    estimator=xgboost_model,
    param_grid=xgb_param_grid,
    scoring='accuracy',
    cv=4,
    verbose=1
)
grid_xgboost.fit(X_train_scaled, Y_train)
print("Best parameters found:", grid_xgboost.best_params_)
print("Best Accuracy Score:", grid_xgboost.best_score_)


# ## Randomized Search CV

# In[44]:


xgb_model = xgb.XGBClassifier(objective = 'binary:logistic',
                              subsample= 0.7,
                              n_estimators= 200,
                              max_depth = 9,
                              learning_rate = 0.11,
                              colsample_bytree=0.8)
xgb_model.fit(X_train_scaled, Y_train)


# In[45]:


Y_pred = xgb_model.predict(X_test_scaled)


# # MODEL EVALUATION
# 
# 

# In[46]:


plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(xgb_model, X_test_scaled, Y_test, cmap='Reds', values_format='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[47]:


report = classification_report(Y_test, Y_pred)
print(report)


# # CONCLUSION

# - Both Random Forest and XGBoost models exhibit comparable performance metrics, including accuracy, precision, recall, and F1-score.
# - However, the XGBoost model demonstrates a slightly lower number of false positives and false negatives compared to the Random Forest model.
# - This suggests that the XGBoost model outperforms the Random Forest model slightly in terms of minimizing classification errors.

# THIS PROJECT IS DONE BY RAHUL ASHWANTH

# In[ ]:




