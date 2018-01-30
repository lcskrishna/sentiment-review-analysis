
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import nltk
import string
#get_ipython().magic('matplotlib inline')
#import matplotlib.pyplot as plt
import numpy as np
#import scipy.sparse as sparse

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.metrics import confusion_matrix
from sklearn import metrics
from nltk.corpus import stopwords
from sklearn.neighbors import KNeighborsClassifier


# In[2]:

reviews = pd.read_csv('amazon_baby_train.csv')
reviews.shape
reviews = reviews.dropna()
reviews.shape
print("Done loading the Training Data.")

scores = reviews['rating']
reviews['rating'] = reviews['rating'].apply(lambda x: 1 if x >= 3 else 0)

print("The Mean of the Rating Attribute is : ")
print(scores.mean())
print("The Standard Deviation for the Rating Attribute is : ")
print(scores.std())


# In[3]:

#reviews.groupby('rating')['review'].count()


# In[4]:

#reviews.groupby('rating')['review'].count().plot(kind='bar', color= ['r','g'],title='Label Distribution',  figsize = (10,6))


# In[5]:

def splitPosNeg(Summaries):
    neg = reviews.loc[Summaries['rating']== 0]
    pos = reviews.loc[Summaries['rating']== 1]
    return [pos,neg]
    


# In[6]:

[pos,neg] = splitPosNeg(reviews)


# In[7]:

## Pre Processing Steps.
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

pos_data = []
neg_data = []
for p in pos['review']:
    pos_data.append(preprocessing(p))

for n in neg['review']:
    neg_data.append(preprocessing(n))
print("Done seperating positive and negative reviews.")


# In[9]:

training_data = pos_data + neg_data
training_labels = np.concatenate((pos['rating'].values,neg['rating'].values))
print(training_labels)


# In[10]:

### Splitting the Datasets into 4 Folds to avoid any Memory Errors that could occur while classifying the data.
import math
data_len = len(training_data)
print(data_len)
splitPercentage = 0.25
len_train = math.floor(data_len * splitPercentage);

X_train1 = training_data[:len_train]
Y_train1 = training_labels[:len_train]
print(len(X_train1))

X_train2 = training_data[len_train:len_train+len(X_train1)]
Y_train2= training_labels[len_train:len_train+ len(Y_train1)]
print(len(X_train2))

X_train3 = training_data[len_train+len(X_train1):len_train+len(X_train1)+len(X_train2)]
Y_train3= training_labels[len_train+len(Y_train1):len_train+len(Y_train1)+len(Y_train2)]
print(len(X_train3))


X_train4 = training_data[len_train+len(X_train1)+len(X_train2):data_len]
Y_train4= training_labels[len_train+len(Y_train1)+len(Y_train2):data_len]
print(len(X_train4))


# In[11]:

train_tokens1 = []
train_tokens2 = []
train_tokens3 = []
train_tokens4 = []

for line in X_train1:
    l = nltk.word_tokenize(line)
    for word in l:
        train_tokens1.append(word)
print("Done with tokenizing the reviews for Train Fold 1")

for line in X_train2:
    l = nltk.word_tokenize(line)
    for word in l:
        train_tokens2.append(word)
print("Done with tokenizing the reviews for Train Fold 2")

for line in X_train3:
    l = nltk.word_tokenize(line)
    for word in l:
        train_tokens3.append(word)
print("Done with tokenizing the reviews for Train Fold 3")

for line in X_train4:
    l = nltk.word_tokenize(line)
    for word in l:
        train_tokens4.append(word)
print("Done with tokenizing the reviews for Train Fold 4")



# In[12]:

word_features1 = nltk.FreqDist(train_tokens1)
print("The length of the word features 1 we obtained:")
print(len(word_features1))

word_features2 = nltk.FreqDist(train_tokens2)
print("The length of the word features 2 we obtained:")
print(len(word_features2))

word_features3 = nltk.FreqDist(train_tokens3)
print("The length of the word features 3 we obtained:")
print(len(word_features3))

word_features4 = nltk.FreqDist(train_tokens1)
print("The length of the word features 4 we obtained:")
print(len(word_features4))


# In[13]:

topwords1 = [fpair[0] for fpair in list(word_features1.most_common(4000))]
topwords2 = [fpair[0] for fpair in list(word_features2.most_common(4000))]
topwords3 = [fpair[0] for fpair in list(word_features3.most_common(4000))]
topwords4 = [fpair[0] for fpair in list(word_features4.most_common(4000))]


# In[14]:

## Data formation of the Words.
word_his = pd.DataFrame(word_features1.most_common(20), columns = ['words','count'])
print(word_his)


# In[28]:

vec = CountVectorizer()
c_fit1 = vec.fit_transform([' '.join(topwords1)])
c_fit2 = vec.fit_transform([' '.join(topwords2)])
c_fit3 = vec.fit_transform([' '.join(topwords3)])
c_fit4 = vec.fit_transform([' '.join(topwords4)])
print(c_fit1.shape)
print(c_fit2.shape)
print(c_fit3.shape)
print(c_fit4.shape)


# In[29]:

tf_vec = TfidfTransformer()
tf_fit1 = tf_vec.fit_transform(c_fit1)
tf_fit2 = tf_vec.fit_transform(c_fit2)
tf_fit3 = tf_vec.fit_transform(c_fit3)
tf_fit4 = tf_vec.fit_transform(c_fit4)
print(tf_fit1.shape)
print(tf_fit2.shape)
print(tf_fit3.shape)
print(tf_fit4.shape)


# In[30]:

ctr_features1 = vec.transform(X_train1)
tr_features1 = tf_vec.transform(ctr_features1)
print("Done with forming the Training Features for Fold 1")

ctr_features2 = vec.transform(X_train2)
tr_features2 = tf_vec.transform(ctr_features2)
print("Done with forming the Training Features for Fold 2")

ctr_features3 = vec.transform(X_train3)
tr_features3 = tf_vec.transform(ctr_features3)
print("Done with forming the Training Features for Fold 3")

ctr_features4 = vec.transform(X_train4)
tr_features4 = tf_vec.transform(ctr_features4)
print("Done with forming the Training Features for Fold 4")


# In[31]:

print(tr_features1.shape)
print(Y_train1.shape)
print(tr_features2.shape)
print(Y_train2.shape)
print(tr_features3.shape)
print(Y_train3.shape)
print(tr_features4.shape)
print(Y_train4.shape)


# In[39]:

clf = KNeighborsClassifier(n_neighbors= 6)
clf1 = clf.fit(tr_features1,Y_train1)
clf2 = clf.fit(tr_features2,Y_train2)
clf3 = clf.fit(tr_features3,Y_train3)
clf4 = clf.fit(tr_features4,Y_train4)
print("Done with the classification using KNN with K=6.")


# In[48]:

num_correct1 = 0;
len_tr1 = tr_features1.shape[0]
print(len_tr1)
for i in range(0,len_tr1):
    output_prediction_train = clf1.predict(tr_features1[i])
    if(output_prediction_train[0]== Y_train1[i]):
        num_correct1 = num_correct1 + 1;        

num_correct2= 0;
len_tr2 = tr_features2.shape[0]
for i in range(0,len_tr2):
    output_prediction_train = clf2.predict(tr_features2[i])
    if(output_prediction_train[0]== Y_train2[i]):
        num_correct2 = num_correct2 + 1;        

num_correct3 = 0;
len_tr3 = tr_features3.shape[0]
for i in range(0,len_tr3):
    output_prediction_train = clf3.predict(tr_features3[i])
    if(output_prediction_train[0]== Y_train3[i]):
        num_correct3 = num_correct3 + 1;        

num_correct4 = 0;
len_tr4 = tr_features4.shape[0]
for i in range(0,len_tr4):
    output_prediction_train = clf4.predict(tr_features4[i])
    if(output_prediction_train[0]== Y_train4[i]):
        num_correct4 = num_correct4 + 1;        
        
accuracy_train = (num_correct1 + num_correct2 + num_correct3 + num_correct4)/(len_tr1+len_tr2+len_tr3+len_tr4)
print("Accuracy on the Training Dataset is:")
print(accuracy_train*100)


# In[50]:

## Formation of the Testing Dataset.
reviews = pd.read_csv('amazon_baby_test.csv')
reviews.shape
reviews = reviews.dropna()
reviews.shape
scores = reviews['rating']
reviews['rating'] = reviews['rating'].apply(lambda x: 1 if x >= 3 else 0)
print("Done with loading the Test Dataset.")
print("Mean of the ratings is : ")
print(scores.mean())
print("Standard Deviation of the ratings is: ")
print(scores.std())


# In[51]:

#reviews.groupby('rating')['review'].count()


# In[52]:

#reviews.groupby('rating')['review'].count().plot(kind='bar', color= ['r','g'],title='Label Distribution',  figsize = (10,6))


# In[53]:

[pos,neg] = splitPosNeg(reviews)


# In[54]:

pos_data = []
neg_data = []
for p in pos['review']:
    pos_data.append(preprocessing(p))

for n in neg['review']:
    neg_data.append(preprocessing(n))
print("Done splitting the positives and negatives in the testing dataset")


# In[55]:

testing_data = pos_data + neg_data
testing_labels = np.concatenate((pos['rating'].values,neg['rating'].values))


# In[56]:

print(len(testing_data))
print(len(testing_labels))


# In[57]:

t = []
for line in testing_data:
    l = nltk.word_tokenize(line)
    for w in l:
        t.append(w)
print("Done with tokenizing the Testing dataset.")


# In[58]:

test_word_features = nltk.FreqDist(t)
print(len(test_word_features))


# In[81]:

test_topwords = [fpair[0] for fpair in list(test_word_features.most_common(3999))]


# In[82]:

word_his = pd.DataFrame(test_word_features.most_common(20), columns = ['words','count'])
print(word_his)


# In[83]:

len(test_topwords)


# In[84]:

vec = CountVectorizer()
c_fit = vec.fit_transform([' '.join(test_topwords)])


# In[85]:

tf_vec = TfidfTransformer()
tf_fit = tf_vec.fit_transform(c_fit)


# In[86]:

test_features = vec.transform(testing_data)
test_features = tf_vec.transform(test_features)


# In[87]:

print(test_features.shape)
print(len(testing_labels))


# In[93]:

num_correct1 = 0;
len_test = test_features.shape[0]
print(len_test)
for i in range(0,len_test):
    output_prediction_train = clf1.predict(test_features[i])
    if(output_prediction_train[0]== testing_labels[i]):
        num_correct1 = num_correct1 + 1;        

accuracy_test = (num_correct1)/len_test
print("Accuracy on the Testing Dataset is:")
print(accuracy_test*100)


# In[ ]:



