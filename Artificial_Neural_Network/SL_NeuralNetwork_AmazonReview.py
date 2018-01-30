
#### Assignment2
#### This script includes the implementation of the Neural Networks for the Amazon Sentiment Analysis Dataset.
#### It takes the following libararies - NLTK, pandas, numpy, string, matplotlib, scipy, sklearn.
### @Author: Chaitanya Sri Krishna Lolla, Student ID: 800960353

### Libraries.
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
#from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from nltk.corpus import stopwords
from sklearn.neural_network import MLPClassifier


# In[2]:

### Reading the Given Training Dataset.

reviews = pd.read_csv('amazon_baby_train.csv')
reviews.shape
reviews = reviews.dropna()
reviews.shape
#print(reviews.head(25))

### Converting the Training dataset to follow binary classification with only positive and negative.
### Here the rating above three is considered positive and less than or equal to three are considered as negative cases.
scores = reviews['rating']
reviews['rating'] = reviews['rating'].apply(lambda x: '1' if x >= 3 else '0')
#print(reviews.head(25))

### Mean calculation of the Rating .
print("The Mean of the Rating Attribute is : ")
print(scores.mean())


# In[3]:

## Distribution of the Ratings and its type.
reviews.groupby('rating')['review'].count()


# In[4]:

### Distribution plot of the Ratings.
#reviews.groupby('rating')['review'].count().plot(kind='bar', color= ['r','g'],title='Label Distribution',  figsize = (10,6))


# In[5]:

### This function splits the rating into positive or negative and returns the indices of them in the given array.
def splitPosNeg(Summaries):
    neg = reviews.loc[Summaries['rating'] == '0']
    pos = reviews.loc[Summaries['rating'] == '1']
    return [pos,neg]
    


# In[6]:

[pos,neg] = splitPosNeg(reviews)


# In[7]:

##This step includes pre processing of the review data with the NLP Tool kit .
## This Lemmitizes (Noun, Verb etc of same word is considered as one word) the word
## stems the unnecessary words removing the punctuation.

#stemmer = PorterStemmer()
lemmatizer = nltk.WordNetLemmatizer()
stop = stopwords.words('english')
translation = str.maketrans(string.punctuation,' '*len(string.punctuation))

#filtered_words = [word for word in word_list if word not in stopwords.words('english')]

def preprocessing(line):
    tokens=[]
    line = line.translate(translation)
    line = nltk.word_tokenize(line.lower())
    #print(line)
    stops = stopwords.words('english')
    stops.remove('not')
    stops.remove('no')
    line = [word for word in line if word not in stops]
    #print("After removing stop words")
    #print(line)
    for t in line:
        #if(t not in stop):
            #stemmed = stemmer.stem(t)
        stemmed = lemmatizer.lemmatize(t)
        tokens.append(stemmed)
    return ' '.join(tokens)


# In[8]:

### Splitting the positive and negative words using the preprocessing method we have written.
pos_data = []
neg_data = []
for p in pos['review']:
    pos_data.append(preprocessing(p))

for n in neg['review']:
    neg_data.append(preprocessing(n))
print("Done")


# In[9]:

### Concatenating the positive and negative data into a single array and also the labels corrosponding to them.
data = pos_data + neg_data
labels = np.concatenate((pos['rating'].values,neg['rating'].values))
#print(labels)


# In[10]:

### Tokenize the words in the given training data.
t = []
for line in data:
    l = nltk.word_tokenize(line)
    for w in l:
        t.append(w)
#print(t)


# In[11]:

### Identifying the counts of the words.
word_features = nltk.FreqDist(t)
print(len(word_features))


# In[12]:

## Displaying what are the counts of the topwords .
topwords = [fpair[0] for fpair in list(word_features.most_common(5000))]
print(word_features.most_common(25))


# In[13]:

## Checking what are the most common words used from the above.
word_his = pd.DataFrame(word_features.most_common(200), columns = ['words','count'])
#print(word_his)


# In[14]:

### Count Vectorizer is used to convert the text into a matrix that contains the count of each word.
vec = CountVectorizer()
c_fit = vec.fit_transform([' '.join(topwords)])


# In[15]:

### TFID Transformer is used for normalization of the above matrix we have formed using the topwords.
tf_vec = TfidfTransformer()
tf_fit = tf_vec.fit_transform(c_fit)


# In[16]:

#### This step is used for forming the training features that we are going to give to our classification problem.
ctr_features = vec.transform(data)
tr_features = tf_vec.transform(ctr_features)


# In[17]:

tr_features.shape


# In[18]:

### Applying Multi Layer Perceptron (Neural Networks) classification on the above training features formed.
### Prediction on the training dataset and accuracy calculation.
### The Default Parameters of Iteration = 600, solver = "sgd'- Stochastic Gradient Descent Algorithm, Hidden layers = 600, and learning rate =  0.001, activation layer = 'relu'
### has been used for the classification.
clf = MLPClassifier(hidden_layer_sizes = (600,), activation='relu', solver='sgd', alpha = 0.0001, verbose=True, learning_rate = 'constant',learning_rate_init= 0.001,max_iter=500)
clf = clf.fit(tr_features, labels)
tfPredication = clf.predict(tr_features)
tfAccuracy = metrics.accuracy_score(tfPredication,labels)
print(tfAccuracy * 100)


# In[19]:

#### Testing Dataset and its prediction following the above same procedure.
reviews = pd.read_csv('amazon_baby_test.csv')
reviews.shape
reviews = reviews.dropna()
reviews.shape
#print(reviews.head(25))

scores = reviews['rating']
reviews['rating'] = reviews['rating'].apply(lambda x: '1' if x > 3 else '0')
#print(reviews.head(25))


scores.mean()


# In[20]:

reviews.groupby('rating')['review'].count()


# In[21]:

#reviews.groupby('rating')['review'].count().plot(kind='bar', color= ['r','g'],title='Label Distribution',  figsize = (10,6))


# In[22]:

[pos,neg] = splitPosNeg(reviews)


# In[23]:

pos_data = []
neg_data = []
for p in pos['review']:
    pos_data.append(preprocessing(p))

for n in neg['review']:
    neg_data.append(preprocessing(n))
print("Done")


# In[24]:

data = pos_data + neg_data
labels = np.concatenate((pos['rating'].values,neg['rating'].values))
#print(labels)


# In[25]:

t = []
for line in data:
    l = nltk.word_tokenize(line)
    for w in l:
        t.append(w)
#print(t)


# In[26]:

word_features = nltk.FreqDist(t)
print(len(word_features))


# In[41]:

topwords = [fpair[0] for fpair in list(word_features.most_common(5002))]
print(word_features.most_common(25))


# In[42]:

word_his = pd.DataFrame(word_features.most_common(200), columns = ['words','count'])
#print(word_his)


# In[43]:

vec = CountVectorizer()
c_fit = vec.fit_transform([' '.join(topwords)])


# In[44]:

tf_vec = TfidfTransformer()
tf_fit = tf_vec.fit_transform(c_fit)


# In[45]:

cte_features = vec.transform(data)
te_features = tf_vec.transform(cte_features)


# In[46]:

### Prediction of the Given Testing dataset and its accuracy.
tePredication = clf.predict(te_features)
teAccuracy = metrics.accuracy_score(tePredication,labels)
print(teAccuracy)


# In[47]:

te_features.shape


# In[48]:

### Printing the various metrics used for accuracy on Testing dataset - F1 Score, Recall,Precision. 
print(metrics.classification_report(labels, tePredication))


# In[ ]:



