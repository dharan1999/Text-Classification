# -*- coding: utf-8 -*-

import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files
nltk.download('stopwords')

#import datasets
reviews = load_files('txt_sentoken/')

x,y = reviews.data,reviews.target

#storing as pickle files
with open('x.pickle','wb') as f:
    pickle.dump(x,f)
    
with open('y.pickle','wb') as f:
    pickle.dump(y,f)
    
#unpickling datasets
    
with open('x.pickle','rb') as f:
    x = pickle.load(f)
    
with open('y.pickle','rb') as f:
    y = pickle.load(f)
    
#preprocess the data
corpus = []
for i in range(0,len(x)):
    review = re.sub(r'\W',' ',str(x[i]))
    review = review.lower()
    review = re.sub(r'\s+[a-z]\s+',' ',review)
    review = re.sub(r'^[a-z]\s+',' ',review)
    review = re.sub(r'\s+',' ',review)
    corpus.append(review)
    
    
#bag of words model
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features = 2000,min_df = 3,max_df = 0.6,stop_words = stopwords.words('english'))
x = vectorizer.fit_transform(corpus).toarray()


#TFIDF model
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
x = transformer.fit_transform(x).toarray()

#to create vectorizer, for storring only one file
    

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 2000,min_df = 3,max_df = 0.6,stop_words = stopwords.words('english'))
x = vectorizer.fit_transform(corpus).toarray()


#splitting the data for training and testing
from sklearn.model_selection import train_test_split
text_train,text_test,sent_train,sent_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
    

#training the classifier

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(text_train,sent_train)

#Testing the performance
sent_pred = classifier.predict(text_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(sent_test,sent_pred)
 
#pickling the classifier
with open('classifier.pickle','wb') as f:
    pickle.dump(classifier,f)


#pickling the vectorizer
with open('Tfidfmodel.pickle','wb') as f:
    pickle.dump(vectorizer,f)

#unpickling the classifier and vectorier
with open('classifier.pickle','rb') as f:
    clf = pickle.load(f)
    
with open('Tfidfmodel.pickle','rb') as f:
    tfidf = pickle.load(f)
    
sample = ['You are a nice person man, have a good life']
sample = tfidf.transform(sample).toarray()
print(clf.predict(sample))   



