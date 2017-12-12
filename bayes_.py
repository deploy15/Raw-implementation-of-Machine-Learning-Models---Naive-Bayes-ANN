#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from math import log
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os


def initializers(X_train,y_train):
        temp_loc=[] # Create a list of temporary Loacation
        each_class_counters = {} #Store the classes
        total_features = {} # Store the total number of features
        temp_loc.append(np.unique(y_train))   # Get all the unique class label into the temporary Location
        classSize =  temp_loc[0].size #Get the number of class size and the number of column 
        featureSize = X_train[0].size #Get the number of features size and the number of column 
        return each_class_counters, total_features, temp_loc, classSize, featureSize

class MyBayesClassifier():
    # For graduate and undergraduate students to implement Bernoulli Bayes
  
    def __init__(self, alpha):
        self._smooth = alpha # This is for add one smoothing, don't forget!
        self._feat_prob = []
        self._class_prob = []
        self._Ncls = []
        self._Nfeat = []
        

    def train(self, X, y):
        # Your code goes here.
        #initialiser
        each_class_counters, total_features, temp_loc, classSize, featureSize = initializers(X,y)
        self._Ncls.append(classSize) #append the total number of classes
        self._Nfeat.append(featureSize)  # append the total number of features
        
        for each_class in range(y.size):
            if y[each_class] in total_features:
                continue
            else:
                total_features[y[each_class]] = [0 for k in range (X[each_class].size)]
                
#feature counter per each class across train, count occurance of each class across train
        for i in range (y.size):
            if y[i] in each_class_counters:
                each_class_counters[y[i]] +=1
            else:
                each_class_counters[y[i]] = 1
            for j in  range(X[i].size):
                    total_features[y[i]][j] += X[i][j]
                    
        #Compute the probability  of class and features for each samples 
        for i in total_features:
            # In the theory of addition smothing to mitigate zero frequency problem, 1 must be added to each number of occurency
            alpha_parameter = self._smooth # smoothing factor
            num_of_occurence = (alpha_parameter + each_class_counters[i])
            total_class_occurence = (y.size+(self._Ncls[0]*alpha_parameter))
            
            self._class_prob.append((num_of_occurence/float(total_class_occurence)))
            res = np.array([])
            for j in  range(X[i].size):
                
                num_of_occurence= (total_features[i][j] + alpha_parameter)
                total_occurence = (each_class_counters[i]+(2*alpha_parameter))
                res =np.append(res,(num_of_occurence/float(total_occurence)))
            self._feat_prob.append(res)
        return 





    def predict(self, X):
        # This is just a place holder so that the code still runs.
        # Your code goes here.
            
        Y_predict = np.array([])

        for i in X:
            assumed_cat = 0  # The category is assumed to be 0
            minusLogProbability = 0 # Initialize the negative logarithm value
            minimumNegativeLogProb = 99999
           
                
            for classValue in range(self._Ncls[0]):
                minusLogProbability = -log(self._class_prob[classValue])
                for featureValue in  range(self._Nfeat[0]):  
                    if ((i[featureValue])==0):
                        minusLogProbability = minusLogProbability - log(1-self._feat_prob[classValue][featureValue])
                    else:
                        minusLogProbability = minusLogProbability - log(self._feat_prob[classValue][featureValue])
                        
                if minimumNegativeLogProb > minusLogProbability:
                    assumed_cat=classValue
                    minimumNegativeLogProb=minusLogProbability
            
            Y_predict=np.append(Y_predict,assumed_cat)
         
        return Y_predict

class MyMultinomialBayesClassifier():
    # For graduate students only
    def __init__(self, smooth=1):
        self._smooth = smooth # This is for add one smoothing, don't forget!
        self._feat_prob = []
        self._class_prob = []
        self._Ncls = []
        self._Nfeat = []

    # Train the classifier using features in X and class labels in Y
    def train(self, X, y):
        # Your code goes here.
        #initialiser
        each_class_counters, total_features, temp_loc, classSize, featureSize = initializers(X,y)
        self._Ncls.append(classSize) #append the total number of classes
        self._Nfeat.append(featureSize)  # append the total number of features
        
        for each_class in range(y.size):
            if y[each_class] in total_features:
                continue
            else:
                total_features[y[each_class]] = [0 for k in range (X[each_class].size)]
                
#feature counter per each class across train, count occurance of each class across train
        for i in range (y.size):
            if y[i] in each_class_counters:
                each_class_counters[y[i]] +=1
            else:
                each_class_counters[y[i]] = 1
            for j in  range(X[i].size):
                    total_features[y[i]][j] += X[i][j]
                    
        #Compute the probability  of class and features for each samples 
       
        self._class_prob.append(each_class_counters)
        self._feat_prob.append(total_features)
        return

    # should return an array of predictions, one for each row in X
    def predict(self, X):
        # This is just a place holder so that the code still runs.
        # Your code goes here.
        
        Y_predict = np.array([])
        #Total Training Counter
        totalTrainingCounter = 0
        for key in self._class_prob[0]:
            totalTrainingCounter += self._class_prob[0][key]
        
        
        for i in X:
            assumed_cat = 0  # The category is assumed to be 0
            minusLogProbability = 0 # Initialize the negative logarithm value
            minimumNegativeLogProb = 99999
            
            for classValue in self._feat_prob[0]:
                totalFeatureProb = sum(self._feat_prob[0][classValue])
                minusLogProbability = -log((self._class_prob[0][classValue]+1)/float(totalTrainingCounter+(self._Ncls[0]*self._smooth)))
                for j in  range(self._Nfeat[0]):  
                # For multinomial we dont consider value = 0, so we continue iteration to save Computation time
                    if ((i[j])==0):
                        continue    
                    for k in range (i[j]):
                        num_of_occurence = (self._smooth+self._feat_prob[0][classValue][j])
                        total_class_occurence = (totalFeatureProb+(self._Nfeat[0]*self._smooth))
                        minusLogProbability -= log(num_of_occurence/float(total_class_occurence))
                        
                if minimumNegativeLogProb>minusLogProbability:
                    assumed_cat=classValue
                    minimumNegativeLogProb=minusLogProbability
            
            Y_predict=np.append(Y_predict,assumed_cat)
         
        return Y_predict



# generate a random Number
rndList = []
def generateRandomValue(initialValue, finalValue,step): 
    rndList.append(0.01)
    while (initialValue < finalValue):
        initialValue +=  step
        rndList.append(initialValue)
    return rndList

def save_to_fileMNNB(path,textToWrite):
    with open(path +'/MultinomialNB.csv', mode='w+', encoding='utf-8') as myfile:
        myfile.write('\n'.join(textToWrite))
        myfile.write('\n')


def save_to_fileNB(path,textToWrite):
    with open(path +'/NB.csv', mode='w+', encoding='utf-8') as myfile:
        myfile.write('\n'.join(textToWrite))
        myfile.write('\n')

      
categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
remove = ('headers', 'footers', 'quotes')

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)
print('data loaded')

y_train, y_test = data_train.target, data_test.target



print("Extracting features from the training data using a count vectorizer")

# Binary = true for Bernoulli NB
vectorizer = CountVectorizer(stop_words='english', binary=True)#, analyzer='char', ngram_range=(1,3))
X_train = vectorizer.fit_transform(data_train.data).toarray()
X_test = vectorizer.transform(data_test.data).toarray()
feature_names = vectorizer.get_feature_names()

# For Bernoulli NB, Binary = true, train for one default smooth value alpha = 1


print ('******* Question 3a. Additive Smoothing with Alpha = 1  ***********')
alpha = 1
clf = MyBayesClassifier(alpha)
clf.train(X_train,y_train)
y_pred = clf.predict(X_test)

print (" Bernoulli NB with alpha = 1:  " +'accuracy = %f' %(np.mean((y_test-y_pred)==0)))

print ('************    END of Question 3a *****************************')





print ('*** Question 3b  With Varing Degree of Alpha   ***')

#print (generateRandomValue(0.01,1,0.01))
#print (len(generateRandomValue(0.01,1,0.01)))
alphaList = generateRandomValue(0.01,1,0.01)
accuracyList = []
accuracyAlphaListNB = []
for alpha in  alphaList:
    clf = MyBayesClassifier(alpha)
    clf.train(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = (np.mean((y_test-y_pred)==0))
    accuracyList.append(accuracy)
    accuracyAlphaListNB.append(str(alpha)+","+ str(accuracy))
    
    print (" Bernoulli NB  " +'alpha=%f ,accuracy = %f' %(alpha, np.mean((y_test-y_pred)==0)))
    
    print ('******************************************************')

plt.plot(alphaList,accuracyList)
plt.title("Accuracy Plots Using Additive Smoothing with Bernoulli NB")
plt.ylabel("Accuracy")
plt.xlabel("Alpha")
plt.legend()
plt.show()

path = os.path.dirname(os.path.abspath(__file__))
#print (path)
save_to_fileMNNB(path, accuracyAlphaListNB)

with PdfPages('BernoulliNB.pdf') as pdf:
    plt.plot(alphaList,accuracyList,marker='.', linestyle='-', color='r')
    plt.ylabel('Accuracy',color='g')
    plt.xlabel('Alpha',color='g')
    plt.title('BernoulliNB with Accuracy Against Alpha',color = 'k')
    pdf.savefig() 
    plt.close()




print ('*** Question 3c  With Varing Degree of Alpha for MultinomialNB  ***')

print ("Extracting data with Binary =False for Multinomial NB")
vectorizer = CountVectorizer(stop_words='english', binary=False)#, analyzer='char', ngram_range=(1,3))
X_train = vectorizer.fit_transform(data_train.data).toarray()
X_test = vectorizer.transform(data_test.data).toarray()
feature_names = vectorizer.get_feature_names()



alphaList = generateRandomValue(0.01,1,0.01)
accuracyList = []
accuracyAlphaList = []
for alpha in  alphaList:
    objFit = MyMultinomialBayesClassifier(alpha)
    objFit.train(X_train,y_train)
    y_pred = objFit.predict(X_test)
    accuracy = (np.mean((y_test-y_pred)==0))
    accuracyList.append(accuracy)
    accuracyAlphaList.append(str(alpha)+","+ str(accuracy))
    
    print (" Multinomial NB  " +'alpha=%f ,accuracy = %f' %(alpha, np.mean((y_test-y_pred)==0)))
    
    print ('***************************************')

#plot on the screen
plt.plot(alphaList,accuracyList)
plt.title("Accuracy Plots Using Additive Smoothing with Multinomial NB")
plt.ylabel("Accuracy")
plt.xlabel("Alpha")
plt.legend()
plt.show()

#Write outputs to csv
#print (accuracyAlphaList)

path = os.path.dirname(os.path.abspath(__file__))
#print (path)
save_to_fileMNNB(path, accuracyAlphaList +","+ accuracyAlphaListNB)
#Plot into a PDF
with PdfPages('MultinomialNB.pdf') as pdf:
    plt.plot(alphaList,accuracyList,marker='.', linestyle='-', color='r')
    plt.ylabel('Accuracy',color='g')
    plt.xlabel('Alpha',color='g')
    plt.title('Alpha Vs accuracy plot for MultinomialNB',color = 'k')
    pdf.savefig() 
    plt.close()




'''
#verification
print (X_train[X_train ==1])
print (X_train.shape)
print (X_test)
print (X_test.shape)
print (y_train)
print (y_train.shape)
print (y_test)
print (y_test.shape)
print (len(feature_names))

'''