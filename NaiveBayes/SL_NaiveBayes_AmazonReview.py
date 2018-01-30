
# coding: utf-8

# In[1]:

### This file contains the Naive Bayes Implementation on the Amazon Review Dataset.
### The below contains the libraries used.
### @Author: Chaitanya Sri Krishna Lolla.
import pandas as pd
import numpy as np
import nltk
import string
#get_ipython().magic('matplotlib inline')
#import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from nltk.corpus import stopwords
from sklearn.naive_bayes import GaussianNB


# In[2]:

## Loading of the Training dataset and splitting the classes into two classes positive and negative.
reviews = pd.read_csv('amazon_baby_train.csv')
reviews.shape
reviews = reviews.dropna()
reviews.shape

scores = reviews['rating']
reviews['rating'] = reviews['rating'].apply(lambda x: 1 if x >= 3 else 0)
print("Done formation of the Training dataset.")
## The calculation of the Mean and standard deviation for the Training Classes.
print("The mean of output classes in the Training dataset is:")
print(scores.mean())
print("The standard deviation of the output classes in the Training dataset is:")
print(scores.std())


# In[3]:

### Distribution of the Training Output classes.
#reviews.groupby('rating')['review'].count()


# In[4]:

#reviews.groupby('rating')['review'].count().plot(kind='bar', color= ['r','g'],title='Label Distribution',  figsize = (10,6))


# In[5]:

## This method is responsible for splitting the data into positive and negative reviews.
def splitPosNeg(Summaries):
    neg = reviews.loc[Summaries['rating']== 0]
    pos = reviews.loc[Summaries['rating']== 1]
    return [pos,neg]


# In[6]:

[pos,neg] = splitPosNeg(reviews)


# In[7]:

## Pre Processing Steps which uses lemmitizer and stopwords to clean the reviews.
lemmatizer = nltk.WordNetLemmatizer()
stop = stopwords.words('english')
translation = str.maketrans(string.punctuation,' '*len(string.punctuation))

def preprocessing(line):
    tokens=[]
    line = line.translate(translation)
    line = nltk.word_tokenize(line.lower())
    stops = stopwords.words('english')
    stops.remove('not')
    stops.remove('no')
    line = [word for word in line if word not in stops]
    for t in line:
        stemmed = lemmatizer.lemmatize(t)
        tokens.append(stemmed)
    return ' '.join(tokens)


# In[8]:

### This method actually preprocesses the data.
pos_data = []
neg_data = []
for p in pos['review']:
    pos_data.append(preprocessing(p))

for n in neg['review']:
    neg_data.append(preprocessing(n))
print("Done forming the positive and negative reveiws.")


# In[9]:

### The formation of the Training Data.
training_data = pos_data + neg_data
training_labels = np.concatenate((pos['rating'].values,neg['rating'].values))
print("Done formation of the training features.")


# In[10]:

### This tokenizes the training data using word_tokenize.
tokens = []
for line in training_data:
    l = nltk.word_tokenize(line)
    for w in l:
        tokens.append(w)


# In[11]:

### This tries to find the word features from the training dataset using the frequency distribution.
word_features = nltk.FreqDist(tokens)
print(len(word_features))


# In[12]:

### Identifying the training top words for formation of the sparse matrix.
training_topwords = [fpair[0] for fpair in list(word_features.most_common(5000))]
print(word_features.most_common(25))


# In[13]:

## Printing the top 20 words and its count.
word_his = pd.DataFrame(word_features.most_common(20), columns = ['words','count'])
print(word_his)


# In[14]:

### This method is repsonsible for forming the sparse matrix using the training top words.
vec = CountVectorizer()
c_fit = vec.fit_transform([' '.join(training_topwords)])


# In[15]:

tf_vec = TfidfTransformer()
tf_fit = tf_vec.fit_transform(c_fit)


# In[16]:

## This is responsible for forming the training features using the training data and top words. 
ctr_features = vec.transform(training_data)
training_features = tf_vec.transform(ctr_features)


# In[17]:

print(training_features.shape)


# In[20]:

### Naive Bayes Classification model using Gaussian Naive Bayes Implementation without any priors.
clf = GaussianNB()
training_features = training_features.toarray()
clf = clf.fit(training_features,training_labels)
print("Classification is Done using Gaussian NB.")


# In[ ]:

## Formation of the Errors and Accuracies of Training dataset.
output_Predicted = clf.predict(training_features);
accuracy_training = metrics.accuracy_score(output_Predicted,training_labels)
print(accuracy_training* 100)


# In[ ]:

### This is responsible for loading the Testing dataset.
reviews = pd.read_csv('amazon_baby_test.csv')
reviews.shape
reviews = reviews.dropna()
reviews.shape


scores = reviews['rating']
reviews['rating'] = reviews['rating'].apply(lambda x: 1 if x >= 3 else 0)
print("Done loading the testing dataset.")

print("Mean of the output classes in Testing dataset:")
print(scores.mean())
print("Standard Deviation of the output classes in the Testing dataset is:")
print(scores.std())


# In[ ]:

### Splitting the testing data into postive and negative reviews.
[pos,neg] = splitPosNeg(reviews)


# In[ ]:

pos_data = []
neg_data = []
for p in pos['review']:
    pos_data.append(preprocessing(p))

for n in neg['review']:
    neg_data.append(preprocessing(n))
print("Done forming the positive and negative reviews in testing dataset.")


# In[ ]:

## Formation of the Testing data.
testing_data = pos_data + neg_data
testing_labels = np.concatenate((pos['rating'].values,neg['rating'].values))


# In[ ]:

## Formation of tokesn in testing dataset.
t = []
for line in testing_data:
    l = nltk.word_tokenize(line)
    for w in l:
        t.append(w)


# In[ ]:

## Formation of the word features for the testing dataset.
word_features = nltk.FreqDist(t)
print(len(word_features))


# In[ ]:

topwords = [fpair[0] for fpair in list(word_features.most_common(5002))]
print(word_features.most_common(25))


# In[ ]:

## Printing the count and top words formed.
word_his = pd.DataFrame(word_features.most_common(20), columns = ['words','count'])
print(word_his)


# In[ ]:

### Formation of the sparse matrix
vec = CountVectorizer()
c_fit = vec.fit_transform([' '.join(topwords)])


# In[ ]:

tf_vec = TfidfTransformer()
tf_fit = tf_vec.fit_transform(c_fit)


# In[ ]:

cte_features = vec.transform(testing_data)
te_features = tf_vec.transform(cte_features)


# In[ ]:

te_features.shape


# In[ ]:

te_features = te_features.toarray()
tePredication = clf.predict(te_features)
teAccuracy = metrics.accuracy_score(tePredication,testing_labels)
print("Accuracy of the Testing dataset is:")
print(teAccuracy * 100)


# In[ ]:

# printing the metrics
print(metrics.classification_report(labels, tePredication))

