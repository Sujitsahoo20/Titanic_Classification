#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# In[8]:


os.chdir("D:\\DATASET")


# In[14]:


titanic_data = pd.read_csv("train.csv")


# # Exploratory Data Analysis

# In[15]:


titanic_data.head(10)


# In[16]:


titanic_data.tail(10)


# In[48]:


titanic_data.describe()


# # Visualization

# In[49]:


import seaborn as sns

sns.heatmap(titanic_data.corr(), cmap="YlGnBu")
plt.show()


# In[50]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_indices, test_indices in split.split(titanic_data, titanic_data[["Survived", "Pclass","Sex"]]):
    strat_train_set = titanic_data.loc[train_indices]
    strat_test_set = titanic_data.loc[test_indices]


# In[51]:


plt.subplot(1,2,1)
strat_train_set['Survived'].hist()
strat_train_set['Pclass'].hist()

plt.subplot(1,2,2)
strat_test_set['Survived'].hist()
strat_test_set['Pclass'].hist()

plt.show()


# In[52]:


strat_train_set.info()


# In[53]:


sns.countplot(data=titanic_data,x="Survived")


# In[54]:


sns.barplot(x="Sex", y="Survived", data=titanic_data)


# In[55]:


titanic_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[56]:


titanic_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[57]:


titanic_data['Embarked'].value_counts().plot(kind='bar', rot=0, color=['red', 'blue', 'green'])
plt.title('Embarked')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()


# In[58]:


travelling_partners = titanic_data['SibSp'] + titanic_data['Parch']
travelled_alone = np.where(travelling_partners > 0, 0, 1)

survival_rates = titanic_data.groupby(travelled_alone)['Survived'].mean()

sns.barplot(x=survival_rates.index, y=survival_rates.values)
plt.xlabel('Not Alone (0) vs. Alone (1)')
plt.ylabel('Survival Rate')
plt.xticks([0, 1], ['Not Alone', 'Alone'])
plt.show()


# In[59]:


bins = [0, 12, 18, 100]
labels = ['Children', 'Teenagers', 'Adults']

age_groups = pd.cut(titanic_data['Age'], bins=bins, labels=labels, right=False)
pivot_table = pd.crosstab(index=age_groups, columns=titanic_data['Survived'])

# Plot the graph
ax = pivot_table.plot(kind='bar', stacked=True)
ax.set_xlabel('Age Group')
ax.set_ylabel('Survival Count')
ax.set_title('Survival Count by Age Group')
plt.xticks(rotation=0)
plt.legend(title='Survived', labels=['Not Survived', 'Survived'])
plt.show()


# # Data Cleaning

# In[61]:


titanic_data['Age'].fillna(round(titanic_data['Age'].mean()),inplace=True)
titanic_data['Embarked'].fillna('S',inplace=True)
titanic_data['Cabin'].fillna('C85', inplace=True)
titanic_data.head()


# In[63]:


cat_sex={"male":0,"female":1}
titanic_data["Sex"]=titanic_data["Sex"].map(cat_sex)


# In[64]:


titanic_data.head()


# In[65]:


cat_Embarked={"S":0,"C":1,"Q":2}
titanic_data["Embarked"]=titanic_data["Embarked"].map(cat_Embarked)
titanic_data.head()


# In[66]:


titanic_data.isnull().sum()


# # Model Comparision

# In[67]:


models = ['Random Forest', 'Gradient Boosting', 'Logistic Regression', 'Decision Tree']
accuracies = [78, 83, 67, 76]

plt.bar(models, accuracies, color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 100)  
plt.xticks(rotation=45) 
for i, accuracy in enumerate(accuracies):
    plt.text(i, accuracy + 1, f'{accuracy}%', ha='center', va='bottom', fontsize=10)
plt.tight_layout()  
plt.show()


# In[68]:


models = ['Random Forest', 'Gradient Boosting', 'Logistic Regression', 'Decision Tree']
Precisions = [75, 86, 80, 71]

plt.bar(models, Precisions, color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Models')
plt.ylabel('Precision (%)')
plt.title('Model Precision Comparison')
plt.ylim(0, 100)  
plt.xticks(rotation=45) 
for i, Precision in enumerate(Precisions):
    plt.text(i, Precision + 1, f'{Precision}%', ha='center', va='bottom', fontsize=10)
plt.tight_layout()  
plt.show()


# In[69]:


models = ['Random Forest', 'Gradient Boosting', 'Logistic Regression', 'Decision Tree']
Recall = [72, 69, 26, 70]

plt.bar(models, Recall, color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Models')
plt.ylabel('Recall (%)')
plt.title('Model Recall Comparison')
plt.ylim(0, 100)  
plt.xticks(rotation=45) 
for i, R in enumerate(Recall):
    plt.text(i, R + 1, f'{R}%', ha='center', va='bottom', fontsize=10)
plt.tight_layout()  
plt.show()


# In[ ]:




