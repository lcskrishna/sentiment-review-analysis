### This file contains the Support Vector Classification on the Amazon Dataset.
### The below contains the libraries used.
### @Author: Chaitanya Sri Krishna Lolla.
import pandas as pd
import numpy as np
import nltk
import string
#get_ipython().magic('matplotlib inline')
#import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from nltk.corpus import stopwords
from sklearn import svm


### Loading of the Training dataset and splitting the classes into two classes positive and negative.

reviews = pd.read_csv('amazon_baby_train.csv')
reviews.shape
reviews = reviews.dropna()
reviews.shape

scores = reviews['rating']
reviews['rating'] = reviews['rating'].apply(lambda x: '1' if x >= 3 else '0')

print("The Mean of the Rating Attribute is : ")
print(scores.mean())

print("The standard deviation of the rating column is:")
print(scores.std())


    
### Distribution of the Training Output classes.
reviews.groupby('rating')['review'].count()

reviews.groupby('rating')['review'].count().plot(kind='bar', color= ['r','g'],title='Label Distribution',  figsize = (10,6))


## This method is responsible for splitting the data into positive and negative reviews.

def splitPosNeg(Summaries):
    neg = reviews.loc[Summaries['rating']== '0']
    pos = reviews.loc[Summaries['rating']== '1']
    return [pos,neg]
    

    

# In[7]:

[pos,neg] = splitPosNeg(reviews)


# In[8]:

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


### This method actually preprocesses the data.

pos_data = []
neg_data = []
for p in pos['review']:
    pos_data.append(preprocessing(p))

for n in neg['review']:
    neg_data.append(preprocessing(n))
print("Done")


### The formation of the Training Data.

data = pos_data + neg_data
labels = np.concatenate((pos['rating'].values,neg['rating'].values))
print("Done formation of the training features.")


### This tokenizes the training data using word_tokenize.

t = []
for line in data:
    l = nltk.word_tokenize(line)
    for w in l:
        t.append(w)



### This tries to find the word features from the training dataset using the frequency distribution.

word_features = nltk.FreqDist(t)
print(len(word_features))


### Identifying the training top words for formation of the sparse matrix.

topwords = [fpair[0] for fpair in list(word_features.most_common(5000))]
print(word_features.most_common(25))

## Printing the top 20 words and its count.

word_his = pd.DataFrame(word_features.most_common(200), columns = ['words','count'])



### This method is repsonsible for forming the sparse matrix using the training top words.

vec = CountVectorizer()
c_fit = vec.fit_transform([' '.join(topwords)])


## This is responsible for forming the training features using the training data and top words. 

tf_vec = TfidfTransformer()
tf_fit = tf_vec.fit_transform(c_fit)


# In[17]:

ctr_features = vec.transform(data)
tr_features = tf_vec.transform(ctr_features)


# In[18]:

tr_features.shape


## SVM Classification Implementation. For various results, please check the Results Report.

clf = svm.SVC(kernel='linear')
clf = clf.fit(tr_features, labels)
print("Done Classifying")


# In[ ]:

## Prediction on the Training Dataset.
output_prediction_train = clf.predict(tr_features)
output_train_accuracy = metrics.accuracy_score(output_prediction_train,labels)
print("Accuracy on the Training dataset.")
print(output_train_accuracy * 100)


# In[21]:

## Testing Dataset.
reviews = pd.read_csv('amazon_baby_test.csv')
reviews.shape
reviews = reviews.dropna()
reviews.shape
#print(reviews.head(25))

scores = reviews['rating']
reviews['rating'] = reviews['rating'].apply(lambda x: '1' if x > 3 else '0')
#print(reviews.head(25))


scores.mean()


# In[22]:

reviews.groupby('rating')['review'].count()


# In[23]:

reviews.groupby('rating')['review'].count().plot(kind='bar', color= ['r','g'],title='Label Distribution',  figsize = (10,6))


# In[24]:

[pos,neg] = splitPosNeg(reviews)


# In[25]:

pos_data = []
neg_data = []
for p in pos['review']:
    pos_data.append(preprocessing(p))

for n in neg['review']:
    neg_data.append(preprocessing(n))
print("Done")


# In[26]:

data = pos_data + neg_data
labels = np.concatenate((pos['rating'].values,neg['rating'].values))
#print(labels)


# In[27]:

t = []
for line in data:
    l = nltk.word_tokenize(line)
    for w in l:
        t.append(w)
#print(t)


# In[28]:

word_features = nltk.FreqDist(t)
print(len(word_features))


# In[29]:

topwords = [fpair[0] for fpair in list(word_features.most_common(5002))]
print(word_features.most_common(25))


# In[30]:

word_his = pd.DataFrame(word_features.most_common(200), columns = ['words','count'])
#print(word_his)


# In[31]:

len(topwords)


# In[32]:

vec = CountVectorizer()
c_fit = vec.fit_transform([' '.join(topwords)])


# In[33]:

tf_vec = TfidfTransformer()
tf_fit = tf_vec.fit_transform(c_fit)


# In[34]:

cte_features = vec.transform(data)
te_features = tf_vec.transform(cte_features)


# In[35]:

te_features.shape


# In[36]:

tePredication = clf.predict(te_features)
teAccuracy = metrics.accuracy_score(tePredication,labels)
print(teAccuracy*100)


# In[37]:

print(metrics.classification_report(labels, tePredication))




