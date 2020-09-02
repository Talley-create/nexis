# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 13:42:35 2019

@author: aaron
"""

import pandas as pd
import numpy as np
import nltk.stem

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV

movie_r = pd.read_csv('C:\\Users\\aaron\\Documents\\IST736\\moviereview.tsv', delimiter='\t')

english_stemmer = nltk.stem.SnowballStemmer('english')

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([english_stemmer.stem(w) for w in analyzer(doc)])

review = list(movie_r["text"])
sentiment = list(movie_r["reviewclass"])

   
for x in range(0, len(review)):
    stg_txt = review[x]
    stg_txt=stg_txt.replace("\\n","")
    stg_txt=stg_txt.replace("\\'","")
    stg_txt=stg_txt.replace("\\","")
    stg_txt=stg_txt.replace('"',"")
    stg_txt=stg_txt.replace("'","")
    stg_txt=stg_txt.replace("s'","")
    stg_txt=stg_txt.replace(",","")
    stg_txt=stg_txt.replace(":","")
    stg_txt=stg_txt.replace(";","")
    stg_txt=stg_txt.replace("_","")
    stg_txt=stg_txt.replace("-","")
    stg_txt=stg_txt.replace("(","")
    stg_txt=stg_txt.replace(")","")
    review[x] = stg_txt
    
bi_stem_vec = StemmedCountVectorizer(min_df = 3,
                                  analyzer = 'word',
                                  stop_words = 'english',
                                  lowercase = True,
                                  binary = True,
                                  ngram_range= (2,2))   
uni_stem_vec = StemmedCountVectorizer(min_df = 3,
                                  analyzer = 'word',
                                  stop_words = 'english',
                                  lowercase = True,
                                  binary = True,
                                  ngram_range= (1,1))  
 
    
uniVect = CountVectorizer(input = 'content',
                          analyzer = 'word',
                          stop_words = 'english',
                          lowercase = True,
                          binary = True,
                          ngram_range=(1,1),
                          min_df = 4
                          )

BiVect = CountVectorizer(input = 'content',
                          analyzer = 'word',
                          stop_words = 'english',
                          lowercase = True,
                          binary = True,
                          ngram_range=(2,2),
                          min_df = 3
                          )

uni_v = uniVect.fit_transform(review)
bi_v = BiVect.fit_transform(review)

uni_stem_v = uni_stem_vec.fit_transform(review)
bi_stem_v = bi_stem_vec.fit_transform(review)

Labels = pd.DataFrame(sentiment, columns = ["Labels"])

colnames1 = uniVect.get_feature_names()
colnames2 = BiVect.get_feature_names()
colnames3 = uni_stem_vec.get_feature_names()
colnames4 = bi_stem_vec.get_feature_names()

print(colnames1)

stg_uni = pd.DataFrame(uni_v.toarray(), columns = colnames1) 
stg_bi = pd.DataFrame(bi_v.toarray(), columns = colnames2)

stem_stg_uni = pd.DataFrame(uni_stem_v.toarray(), columns = colnames3)
stem_stg_bi = pd.DataFrame(bi_stem_v.toarray(), columns = colnames4)

uniFinalDF = pd.DataFrame()
biFinalDF = pd.DataFrame()

#uniFinalDF = uniFinalDF.append(stg_uni)
#biFinalDF = biFinalDF.append(stg_bi)

uniFinalDF = uniFinalDF.append(stem_stg_uni)
biFinalDF = biFinalDF.append(stem_stg_bi)


uniFinalDF = uniFinalDF.join(Labels, lsuffix='_caller', rsuffix='_Labels' )  

biFinalDF = biFinalDF.join(Labels, lsuffix='_caller', rsuffix='_Labels' )

uniTrainDF, uniTestDF = train_test_split(uniFinalDF, test_size=0.3)   
biTrainDF, biTestDF = train_test_split(biFinalDF, test_size=0.3)

#Create labels for conf_mat and dataframe for model prediction
uniTrainLabel = uniTrainDF["Labels"]
uniTestLabel = uniTestDF["Labels"]
uniTrain_noLabel = uniTrainDF.drop(["Labels"], axis=1)
uniTest_noLabel = uniTestDF.drop(["Labels"], axis=1)

biTrainLabel = biTrainDF["Labels"]
biTestLabel = biTestDF["Labels"]
biTrain_noLabel = biTrainDF.drop(["Labels"], axis=1)
biTest_noLabel = biTestDF.drop(["Labels"], axis=1)

unifinLabel = uniFinalDF["Labels"]
unifin_noLable = uniFinalDF.drop(["Labels"], axis=1)



uniMultiNB = MultinomialNB()
biMultiNB = MultinomialNB()
uni_svm_model = LinearSVC(C=1.5)
bi_svm_model = LinearSVC(C=1.5)

uniMultiNB.fit(uniTrain_noLabel, uniTrainLabel)
uni_pred = uniMultiNB.predict(uniTest_noLabel)

uni_nb_cf = confusion_matrix(uniTestLabel, uni_pred)
print(uni_nb_cf)

precision_recall_fscore_support(uniTestLabel, uni_pred)
precision_recall_fscore_support(biTestLabel, bi_pred)

biMultiNB.fit(biTrain_noLabel, biTrainLabel)
bi_pred = biMultiNB.predict(biTest_noLabel)

bi_nb_cf = confusion_matrix(biTestLabel, bi_pred)
print(bi_nb_cf)

uni_svm_model.fit(uniTrain_noLabel, uniTrainLabel)
bi_svm_model.fit(biTrain_noLabel, biTrainLabel)
uni_svm_pred = uni_svm_model.predict(uniTest_noLabel)
bi_svm_pred = bi_svm_model.predict(biTest_noLabel)
uni_svm_cf = confusion_matrix(uniTestLabel, uni_svm_pred)
bi_svm_cf = confusion_matrix(biTestLabel, bi_svm_pred)

accuracy_score(uniTestLabel, uni_svm_pred)
accuracy_score(biTestLabel, bi_svm_pred)
accuracy_score(uniTestLabel, uni_pred)
accuracy_score(biTestLabel, bi_pred)

print(uni_svm_cf)
print(uni_nb_cf)
print(bi_svm_cf)
print(bi_nb_cf)

precision_recall_fscore_support(biTestLabel, bi_svm_pred)




print(sorted(uniVect.vocabulary_.items()))

ranked_uni = sorted(zip(uniMultiNB.feature_log_prob_[0], uni_stem_vec.get_feature_names()))


for x in range(0,9):
    print(ranked_uni[x])
    
for y in range((len(ranked_uni)-10), len(ranked_uni)):
    print(ranked_uni[y])
    
print(bi_svm_model.coef_[0])
ranked_uni_svm = sorted(zip(uni_svm_model.coef_[0], uni_stem_vec.get_feature_names()))

for x in range(0,9):
    print(ranked_uni_svm[x])
    
for x in range(len(ranked_uni_svm) - 10, len(ranked_uni_svm)):
    print(ranked_uni_svm[x])
    
ranked_bi_svm = sorted(zip(bi_svm_model.coef_[0], bi_stem_vec.get_feature_names()))

for x in range(0,9):
    print(ranked_bi_svm[x])
    
for x in range(len(ranked_bi_svm) - 10, len(ranked_bi_svm)):
    print(ranked_bi_svm[x])
    
print(uniTrainDF.shape)
svm_clf = svm.SVC(kernel='linear', C=1)    
scores = cross_val_score(svm_clf, uniTrain_noLabel, uniTrainLabel, cv = 10)
print(scores.mean(), scores.std())  

mnb_clf = MultinomialNB()   
scores_nm = cross_val_score(mnb_clf, uniTrain_noLabel, uniTrainLabel, cv = 10)
print(scores_nm.mean(), scores_nm.std())

parms = {'kernel':('linear', 'rbf'), 'C':[1,5]}
svc = svm.SVC(gamma="scale")
clf = GridSearchCV(svc, parms, cv=10)
clf.fit(unifin_noLable, unifinLabel)  
clf.cv_results_    
