#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np                               #importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
from sklearn.naive_bayes import GaussianNB
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv('datasets_33180_43520_heart.csv')


# In[3]:


df.info()


# In[4]:


import seaborn as sns
#get correaltion of each features in dataset
corrmat = df.corr()
top_corr_features= corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g = sns.heatmap(df[top_corr_features].corr(),annot=True,cmap='RdYlGn')


# In[5]:


df.hist()


# In[6]:


sns.set_style('whitegrid')
sns.countplot(x='target',data=df,palette='RdBu_r')


# In[7]:


dataset = pd.get_dummies(df,columns=['sex','cp','fbs','restecg','exang','slope','ca','thal'])


# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
standardscaler = StandardScaler()
columns_to_scale = ['age','trestbps','chol','thalach','oldpeak']
dataset[columns_to_scale]= standardscaler.fit_transform(dataset[columns_to_scale])


# In[9]:


dataset.head()


# In[10]:


y = dataset['target']
X = dataset.drop(['target'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)


# In[11]:


knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X_train, y_train)
    knn_scores.append(knn_classifier.score(X_test, y_test))


# In[12]:


plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')
for i in range(1,21):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1, 21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')


# In[13]:


from sklearn.model_selection import cross_val_score
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print(preds)


# In[14]:


from sklearn.neighbors import KNeighborsClassifier
kn_classifier = KNeighborsClassifier(n_neighbors = 10)
score= cross_val_score(kn_classifier,X,y,cv=10)
score.mean()


# In[15]:


nb_conf_mtr=pd.crosstab(y_test,preds)
nb_conf_mtr


# In[16]:


from sklearn.metrics import classification_report
nbreport = classification_report(y_test,preds)
print(nbreport)


# # Random Forest

# In[17]:


from sklearn.ensemble import RandomForestClassifier
random_forest_classifier = RandomForestClassifier(n_estimators = 10)
score= cross_val_score(random_forest_classifier,X,y,cv=10)
score.mean()


# In[18]:


clfs = RandomForestClassifier()
clfs.fit(X_train, y_train)
predi = clf.predict(X_test)
nb_conf_mtr=pd.crosstab(y_test,predi)
nb_conf_mtr


# In[19]:


from sklearn.metrics import classification_report
nbreport = classification_report(y_test,predi)
print(nbreport)


# # Decision Tree 

# In[20]:


dt_scores = []
for i in range(1, len(X.columns) + 1):
    dt_classifier = DecisionTreeClassifier(max_features = i, random_state = 0)
    dt_classifier.fit(X_train, y_train)
    dt_scores.append(dt_classifier.score(X_test, y_test))


# In[21]:


plt.plot([i for i in range(1, len(X.columns) + 1)], dt_scores, color = 'green')
for i in range(1, len(X.columns) + 1):
    plt.text(i, dt_scores[i-1], (i, dt_scores[i-1]))
plt.xticks([i for i in range(1, len(X.columns) + 1)])
plt.xlabel('Max features')
plt.ylabel('Scores')
plt.title('Decision Tree Classifier scores for different number of maximum features')


# In[22]:


models = DecisionTreeClassifier()
models.fit(X_train,y_train)
test_score=models.score(X_test,y_test)
print("DT Test Score:",test_score)
train_score=models.score(X_train,y_train)
print("DT Train Score:",train_score)


# In[23]:


predis = models.predict(X_test)
nb_conf_mtr=pd.crosstab(y_test,predis)
nb_conf_mtr


# In[24]:


from sklearn.metrics import classification_report
nbreport = classification_report(y_test,predis)
print(nbreport)


# # Naive Bayes Algorithm

# In[25]:


model=GaussianNB()
model.fit(X_train,y_train)


# In[26]:


predict=model.predict(X_test)
predict


# In[27]:


test_score=model.score(X_test,y_test)
print("NB Test Score:",test_score)


# In[28]:


train_score=model.score(X_train,y_train)
print("NB Test Score:",train_score)


# In[29]:


from sklearn.model_selection import cross_validate
cv_results = cross_validate(model,X_train,y_train,cv=10)
cv_results


# In[30]:


nb_conf_mtr=pd.crosstab(y_test,predict)
nb_conf_mtr


# In[31]:


from sklearn.metrics import classification_report
nbreport = classification_report(y_test,predict)
print(nbreport)


# In[ ]:




