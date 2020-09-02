# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:28:21 2020

@author: aaron
"""
import nltk
import pandas as pd
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import WhitespaceTokenizer
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.metrics import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.svm import *
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from quantopian.research import returns, symbols

wrk_s_words = stopwords.words('english')
more_words = ['-lrb-', '-rrb-', '?', '\'s','...','--','-', 'b','n\'t','``', '\'\'', ',','.','-LRB-','-RRB-', '`', "'", ':', ';', 'a', 'i', 'lrb', 'rrb']
wrk_s_words.extend(more_words)

s_words = set(wrk_s_words)


wtk = WhitespaceTokenizer()
train_file = ""
#test_file = ""
train_df = pd.read_csv(train_file, delimiter='\t' )
#test_df = pd.read_csv(test_file, delimiter='\t')


train_set = list(zip(train_df['Phrase'].tolist(),train_df['Sentiment'].tolist()))


old_id = ""
review_lst = []
for i in range(len(train_df)):
    curr_id = train_df.loc[i, "SentenceId"]
    if curr_id != old_id:
        review_lst.append((train_df.loc[i, "Phrase"], train_df.loc[i, "Sentiment"]))
        old_id = curr_id        
   
wrd_Freq = nltk.FreqDist([word.lower() for review in train_set if(len(review[0]) > 50) for word in wtk.tokenize(review[0]) if word not in s_words])        
term_freq = wrd_Freq.most_common(1000)
feat_terms = [word for (word,count)in term_freq]


def measures(g,p):
    labels = list(set(g))
    r_list = []
    p_list = []
    f_list = []
    for lab in labels:
        tp = fp = fn = tn = 0
        for i,val in enumerate(g):
            if val == lab and p[i] == lab: tp += 1
            if val == lab and p[i] != lab: fn += 1
            if val != lab and p[i] == lab: fp += 1
            if val != lab and p[i] != lab: tn += 1
        recall = tp/(tp + fp)
        precision = tp/(tp + fn)
        r_list.append(recall)
        p_list.append(precision)
        f_list.append(2 *(recall * precision)/(recall + precision))
        
    print('\tPrecision\tRecall\t\tF1')
    for i,lab in enumerate(labels):
         print(lab, '\t', "{:10.3f}".format(p_list[i]), \
          "{:10.3f}".format(r_list[i]), "{:10.3f}".format(f_list[i]))

        

def readSubjectivity(path):
    flexicon = open(path, 'r')
    # initialize an empty dictionary
    sldict = { }
    for line in flexicon:
        fields = line.split()   # default is to split on whitespace
        # split each field on the '=' and keep the second part as the value
        strength = fields[0].split("=")[1]
        word = fields[2].split("=")[1]
        posTag = fields[3].split("=")[1]
        stemmed = fields[4].split("=")[1]
        polarity = fields[5].split("=")[1]
        if (stemmed == 'y'):
            isStemmed = True
        else:
            isStemmed = False
        # put a dictionary entry with the word as the keyword
        #     and a list of the other values
        sldict[word] = [strength, posTag, isStemmed, polarity]
    return sldict

sbj_path = 'C:\\Users\\aaron\\Documents\\IST 664\\subjclueslen1_hltemnlp05\\subjclueslen1-HLTEMNLP05.tff'
SL_Reader = readSubjectivity(sbj_path)

def cross_validation_accuracy(num_folds, featuresets):
    subset_size = int(len(featuresets)/num_folds)
    print('Each fold size:', subset_size)
    accuracy_list = []
    # iterate over the folds
    for i in range(num_folds):
        test_this_round = featuresets[(i*subset_size):][:subset_size]
        train_this_round = featuresets[:(i*subset_size)] + featuresets[((i+1)*subset_size):]
        # train using train_this_round
        classifier = nltk.NaiveBayesClassifier.train(train_this_round)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
        gold_lst = []
        pred_lst = []
        for i,(feature, label) in enumerate(test_this_round):
            gold_lst.append(label)
            pred_lst.append(classifier.classify(feature))   
        cm = nltk.ConfusionMatrix(gold_lst,pred_lst)
        print (i, accuracy_this_round)
        measures(gold_lst, pred_lst)
        #print('Recall: ',recall(set(gold_lst),set(pred_lst)))
        print(cm.pretty_format(sort_by_count=True,show_percents=True, truncate=9))
        
        accuracy_list.append(accuracy_this_round)
    # find mean accuracy over all rounds
    print ('mean accuracy', sum(accuracy_list) / num_folds)

def feature_extract(doc, feat):
    #print(doc)
    doc_words = set(doc)
    pos_tag_words = nltk.pos_tag(doc_words)
    features = {}
    
    for word in feat:
        features['contains({})'.format(word)] = (word in doc_words)
    n_noun = 0
    n_verb = 0
    n_adj = 0
    n_adverb = 0
    for (word,tag) in pos_tag_words:
        if tag.startswith('N'): n_noun += 1
        if tag.startswith('V'): n_verb += 1
        if tag.startswith('J'): n_adj += 1
        if tag.startswith('R'): n_adverb += 1
    features['nouns'] = n_noun
    features['verbs'] = n_verb
    features['adjec'] = n_adj
    features['advrb'] = n_adverb
    return features

def SL_features(document, SL, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    weakPos = 0
    strongPos = 0
    weakNeg = 0
    strongNeg = 0
    for word in document_words:
        if word in SL:
            strength, posTag, isStemmed, polarity = SL[word]
            if strength == 'weaksubj' and polarity == 'positive':
                weakPos += 1
            if strength == 'strongsubj' and polarity == 'positive':
                strongPos += 1
            if strength == 'weaksubj' and polarity == 'negative':
                weakNeg += 1
            if strength == 'strongsubj' and polarity == 'negative':
                strongNeg += 1
            features['positivecount'] = weakPos + (2 * strongPos)
            features['negativecount'] = weakNeg + (2 * strongNeg)
    return features

def multi_features(document, SL, feat_terms):
    document_words = set(document)
    #pos_tag_words = nltk.pos_tag(document_words)
    features = {}
    for word in feat_terms:
        features['contains({})'.format(word)] = (word in document_words)
    weakPos = 0
    strongPos = 0
    weakNeg = 0
    strongNeg = 0
    n_noun = 0
    n_verb = 0
    n_adj = 0
    n_adverb = 0
    for word in document_words:
        if word in SL:
            strength, posTag , isStemmed, polarity = SL[word]
            if strength == 'weaksubj' and polarity == 'positive':
                weakPos += 1
            if strength == 'strongsubj' and polarity == 'positive':
                strongPos += 1
            if strength == 'weaksubj' and polarity == 'negative':
                weakNeg += 1
            if strength == 'strongsubj' and polarity == 'negative':
                strongNeg += 1
            if posTag == 'noun': n_noun += 1
            if posTag == 'verb': n_verb += 1
            if posTag == 'adj': n_adj += 1
            if posTag == 'adverb': n_adverb += 1
    
            features['positivecount'] = weakPos + (2 * strongPos)
            features['negativecount'] = weakNeg + (2 * strongNeg) 
        features['nouns'] = n_noun
        features['verbs'] = n_verb
        features['adjec'] = n_adj
        features['advrb'] = n_adverb
    return features




pos_feat = [(feature_extract([word.lower() for word in wtk.tokenize(review) if word not in s_words],feat_terms),cnt ) for (review, cnt) in train_set if (len(review) > 50)]
sub_feat = [(SL_features([word.lower() for word in wtk.tokenize(review) if word not in s_words],SL_Reader, feat_terms),cnt ) for (review, cnt) in train_set if (len(review) > 50)]
multi_feat = [(multi_features([word.lower() for word in wtk.tokenize(review) if word not in s_words],SL_Reader, feat_terms),score ) for (review, score) in train_set if (len(review)> 50)]        
        
cross_validation_accuracy(10, pos_feat)
cross_validation_accuracy(10, sub_feat)
cross_validation_accuracy(10, multi_feat)
 
review = []
cln_review = []
labels = []
for i in range(len(train_df)):
    curr_id = train_df.loc[i, "SentenceId"]
    if curr_id != old_id:
        #print (train_df.loc[i, "Phrase"], train_df.loc[i, "Sentiment"])
        review.append((train_df.loc[i, "Phrase"],train_df.loc[i, "Sentiment"]))
        old_id = curr_id
        
        
for i, (rec,sent) in enumerate(train_set):
    if(len(rec) > 50):
        rec=rec.replace("\\n","")
        rec=rec.replace("\\'","")
        rec=rec.replace("\\","")
        rec=rec.replace('"',"")
        rec=rec.replace("'","")
        rec=rec.replace("s'","")
        rec=rec.replace(",","")
        rec=rec.replace(":","")
        rec=rec.replace(";","")
        rec=rec.replace("_","")
        rec=rec.replace("-","")
        rec=rec.replace("(","")
        rec=rec.replace(")","")
        cln_review.append(rec)
        labels.append(sent)

Lab = pd.DataFrame(labels, columns = ["Labels"])

vec = TfidfVectorizer(stop_words = wrk_s_words,
                      lowercase = True,
                      binary = True,
                      smooth_idf = True)

vec_rev = vec.fit_transform(cln_review)
colnames1 = vec.get_feature_names()

vec_stage = pd.DataFrame(vec_rev.toarray(), columns = colnames1)

FinalDF = vec_stage.join(Lab, lsuffix='_caller', rsuffix='_Labels' ) 

TrainDF, TestDF = train_test_split(FinalDF, test_size=0.3)

TrainLabel = TrainDF["Labels"]
TestLabel = TestDF["Labels"]

Train_noLabel = TrainDF.drop(["Labels"], axis=1)
Test_noLabel = TestDF.drop(["Labels"], axis=1)


svm_model = LinearSVC(C=1)
nm_model = MultinomialNB()

nm_model.fit(Train_noLabel, TrainLabel)
svm_model.fit(Train_noLabel, TrainLabel)

predict_svm = svm_model.predict(Test_noLabel)
predict_nmb = nm_model.predict(Test_noLabel)

cmb = confusion_matrix(TestLabel, predict_nmb)
cm = confusion_matrix(TestLabel, predict_svm)
print(cm)
precision_recall_fscore_support(TestLabel, predict_svm)
accuracy_score(TestLabel, predict_svm)
accuracy_score(TestLabel, predict_nmb)

