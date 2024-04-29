#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, plot_confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC


# In[2]:


import nltk
nltk.download('stopwords')


# In[3]:


# Printing the English stopwords
print(stopwords.words('english'))


# In[4]:


data = pd.read_csv(r"C:\Users\User\Desktop\train.csv")


# In[5]:


print(data)


# In[6]:


data.shape


# # Data pre processing

# In[7]:


data.head(10)


# In[8]:


data.describe()


# In[ ]:





# In[9]:


# Counting the number of missing values in the dataset
data.isnull().sum()


# In[10]:


data = data.fillna('')


# In[11]:


print(data)


# In[11]:


port_stem = PorterStemmer()


# In[12]:


def stemming(text):
  stemmed_content = re.sub('[^a-zA-Z]', ' ', text)
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content = ' '.join(stemmed_content)
  return stemmed_content


# In[13]:


data['text'] = data['text'].apply(stemming)


# In[12]:


print(data['text'])


# # Seperating the data and labels
# 

# In[13]:


X = data['text'].values
Y = data['label'].values


# In[14]:


print(X)


# In[15]:


print(Y)


# In[16]:


#  Convering the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)


# In[17]:


print(X)


# In[18]:


# Splitting the data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state=2)


# # Training the model with logistic regression

# In[19]:


model1 = LogisticRegression()
model1.fit(X_train, Y_train)
predict1 = model1.predict(X_test)


# In[20]:


#Creating a confusion matrix for Logistic Regression
confusion_matrix(Y_test, predict1)


# In[21]:


#Creating the confusion metrics
matrix = classification_report(Y_test, predict1)
print("Classification Report: \n", matrix)


# In[22]:


matrix_plot = plot_confusion_matrix(model1, X_test, Y_test, cmap=plt.cm.Reds)
matrix_plot.ax_.set_title("Confusion Matrix", color='white')
plt.xlabel('Predicted Label', color='black')
plt.ylabel('True Label', color='black')
plt.gcf().axes[0].tick_params(color="white")
plt.gcf().axes[1].tick_params(color="white")
plt.gcf().set_size_inches(10,10)
plt.show()


# In[33]:


false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Y_test, predict1)
print('roc_auc_score for LOGISTIC REGRESSION: ', roc_auc_score(Y_test, predict1))

plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - LR')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # Training the model with SVM

# In[ ]:





# In[34]:


model2 = SVC()
model2.fit(X_train, Y_train)


# In[35]:


predict2 = model2.predict(X_test)


# In[36]:


#Creating a confusion matrix for support vector machine
confusion_matrix(Y_test, predict2)


# In[37]:


#Creating the confusion metrics
matrix = classification_report(Y_test, predict2)
print("Classification Report: \n", matrix)


# In[38]:


matrix_plot = plot_confusion_matrix(model2, X_test, Y_test, cmap=plt.cm.Greens)
matrix_plot.ax_.set_title("Confusion Matrix", color='white')
plt.xlabel('Predicted Label', color='black')
plt.ylabel('True Label', color='black')
plt.gcf().axes[0].tick_params(color="white")
plt.gcf().axes[1].tick_params(color="white")
plt.gcf().set_size_inches(10,10)
plt.show()


# In[39]:


false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Y_test, predict2)
print('roc_auc_score for SUPPORT VECTOR MACHINE: ', roc_auc_score(Y_test, predict2))

plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - SVM')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# #  Training the model with Random Forest

# In[40]:


model3 = RandomForestClassifier()
model3.fit(X_train, Y_train)


# In[41]:


predict3 = model3.predict(X_test)


# In[42]:


# Creating a confusion matrix for Random Forest
confusion_matrix(Y_test, predict3)


# In[43]:


# Creating the confusion metrics
matrix = classification_report(Y_test, predict3)
print("Classification Report : \n", matrix)


# In[44]:


matrix_plot = plot_confusion_matrix(model3, X_test, Y_test, cmap=plt.cm.Blues)
matrix_plot.ax_.set_title("Confusion Matrix", color='white')
plt.xlabel('Predicted Label', color='black')
plt.ylabel('True Label', color='black')
plt.gcf().axes[0].tick_params(color="white")
plt.gcf().axes[1].tick_params(color="white")
plt.gcf().set_size_inches(10,10)
plt.show()


# In[46]:


false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Y_test, predict3)
print('roc_auc_score for RANDOM FOREST: ', roc_auc_score(Y_test, predict3))

plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - RF')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[47]:


from sklearn.preprocessing import LabelEncoder, StandardScaler


# In[57]:


sc = StandardScaler(with_mean=False)


# In[71]:


X_train_standard = sc.fit_transform(X_train)
X_test_standard = sc.transform(X_test)


# # TRAINING THE MODEL WITH LOGISTIC REGRESSION WITH STANSARD SCALAR
# 

# In[72]:


model1 = LogisticRegression()
model1.fit(X_train_standard, Y_train)
predict1 = model1.predict(X_test_standard)


# In[73]:


#Creating the confusion metrics
matrix = classification_report(Y_test, predict1)
print("Classification Report: \n", matrix)


# In[76]:


matrix_plot = plot_confusion_matrix(model1, X_test_standard, Y_test, cmap=plt.cm.Reds)
matrix_plot.ax_.set_title("Confusion Matrix", color='white')
plt.xlabel('Predicted Label', color='black')
plt.ylabel('True Label', color='black')
plt.gcf().axes[0].tick_params(color="white")
plt.gcf().axes[1].tick_params(color="white")
plt.gcf().set_size_inches(10,10)
plt.show()


# In[75]:


false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Y_test, predict1)
print('roc_auc_score for LOGISTIC REGRESSION: ', roc_auc_score(Y_test, predict1))

plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - LR')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # TRAINING THE SVM MODEL WITH STANDARD SCALAR
# 

# In[77]:


model2 = SVC()
model2.fit(X_train_standard, Y_train)


# In[78]:


predict2 = model2.predict(X_test_standard)


# In[82]:


#Creating the confusion metrics
matrix = classification_report(Y_test, predict2)
print("Classification Report: \n", matrix)


# In[84]:


matrix_plot = plot_confusion_matrix(model2, X_test_standard, Y_test, cmap=plt.cm.Greens)
matrix_plot.ax_.set_title("Confusion Matrix", color='white')
plt.xlabel('Predicted Label', color='black')
plt.ylabel('True Label', color='black')
plt.gcf().axes[0].tick_params(color="white")
plt.gcf().axes[1].tick_params(color="white")
plt.gcf().set_size_inches(10,10)
plt.show()


# In[85]:


false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Y_test, predict2)
print('roc_auc_score for SUPPORT VECTOR MACHINE: ', roc_auc_score(Y_test, predict2))

plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - SVM')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # # TRAINING THE RANDOM FOREST MODEL WITH STANDARD SCALAR

# In[87]:


model3 = RandomForestClassifier()
model3.fit(X_train, Y_train)


# In[88]:


predict3 = model3.predict(X_test)


# In[89]:


# Creating a confusion matrix for Random Forest
confusion_matrix(Y_test, predict3)


# In[90]:


# Creating the confusion metrics
matrix = classification_report(Y_test, predict3)
print("Classification Report : \n", matrix)


# In[91]:


matrix_plot = plot_confusion_matrix(model3, X_test, Y_test, cmap=plt.cm.Blues)
matrix_plot.ax_.set_title("Confusion Matrix", color='white')
plt.xlabel('Predicted Label', color='black')
plt.ylabel('True Label', color='black')
plt.gcf().axes[0].tick_params(color="white")
plt.gcf().axes[1].tick_params(color="white")
plt.gcf().set_size_inches(10,10)
plt.show()


# In[45]:


false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Y_test, predict3)
print('roc_auc_score for Naive Bayes FOREST: ', roc_auc_score(Y_test, predict3))

plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - NF')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[32]:


from sklearn.ensemble import RandomForestClassifier


# In[33]:


rf = RandomForestClassifier()


# In[34]:


model4 = rf.fit(X_train, Y_train)


# In[36]:


predict4 = model4.predict(X_test)


# In[37]:


matrix4 = classification_report(predict4, Y_test)


# In[38]:


print(matrix4)


# In[42]:


matrix_plot = plot_confusion_matrix(model4, X_test, Y_test, cmap=plt.cm.Purples)
matrix_plot.ax_.set_title("Confusion Matrix", color='white')
plt.xlabel('Predicted Label', color='black')
plt.ylabel('True Label', color='black')
plt.gcf().axes[0].tick_params(color="white")
plt.gcf().axes[1].tick_params(color="white")
plt.gcf().set_size_inches(10,10)
plt.show()


# In[46]:


false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Y_test, predict4)
print('roc_auc_score for Naive Bayes Forest: ', roc_auc_score(Y_test, predict4))

plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - NB')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:




