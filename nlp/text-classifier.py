import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import os
import json
import datetime

from mlearnlib import sigmoid as sg

'''
TEXT CLASSIFICATION USING NEURAL NETWORKS
Based on gk_'s work at https://machinelearnings.co/text-classification-using-neural-networks-f5cd7b8765c6 and
iamtrask's work at https://iamtrask.github.io

nltk - natural language toolkit
'''

stemmer = LancasterStemmer()

'''CORPUS TRAINING SETUP'''
#create 3 classes of training_text data
training_text = []
#each line represents one document
training_text.append({"class":"greeting", "sentence":"how are you?"})
training_text.append({"class":"greeting", "sentence":"how is your day?"})
training_text.append({"class":"greeting", "sentence":"good day"})
training_text.append({"class":"greeting", "sentence":"good morning"})
training_text.append({"class":"greeting", "sentence":"how is it going today?"})

training_text.append({"class":"goodbye", "sentence":"have a nice day"})
training_text.append({"class":"goodbye", "sentence":"see you later"})
training_text.append({"class":"goodbye", "sentence":"have a nice day"})
training_text.append({"class":"goodbye", "sentence":"talk to you soon"})
training_text.append({"class":"goodbye", "sentence":"till we meet again"})

training_text.append({"class":"command", "sentence":"print me a report"})
training_text.append({"class":"command", "sentence":"can you compile a report?"})
training_text.append({"class":"command", "sentence":"having a difficult time compiling the report?"})
training_text.append({"class":"command", "sentence":"what's the review for?"})
training_text.append({"class":"command", "sentence":"reviewing your email now"})
print ("%s sentences in training_text data" % len(training_text))

words = []
classes = []
documents = []
ignore_words = ['?']

# iterate over each sentence in training_text set
for pattern in training_text:
    # tokenize each word in the sentence
    print(pattern['sentence'])

    w = nltk.word_tokenize(pattern['sentence'])

    # add to our words list
    words.extend(w)

    # add to documents in corpus
    documents.append((w, pattern['class']))

    # add to classes list
    if pattern['class'] not in classes:
        classes.append(pattern['class'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = list(set(words))

# remove duplicates
classes = list(set(classes))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words:", words)

'''TRAINING MODEL SETUP'''

training_data = []
output = []

# create an empty array for the output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0] #doc[1] is the class name

    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

    print(pattern_words, "Pattern Words")

    # create bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

        training_data.append(bag)
    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    output.append(output_row)

# sample training/output
i = 0
w = documents[i][0]
print ([stemmer.stem(word.lower()) for word in w])
print (training_data[i])
print (output[i])

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def is_found_pattern(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

'''
NEURAL NETWORK SETUP
Input: X
Each row is a training example
Each Column is corresponds to neural network's input node
'''

# def think(sentence, show_details=False):
#     x = is_found_pattern(sentence.lower(), words, show_details)
#     if show_details:
#         print ("sentence:", sentence, "\n bag of words:", x)
#     # input layer is our bag of words
#
#     # seed random numbers to make calculation deterministic
#     np.random.seed(1)
#
#     # initialize weights randomly with mean 0
#     syn0 = 2 * np.random.random((3, 1)) - 1
#
#     #process input rows at the same time, i,e full batch training
#
#     for iter in range(10000):
#         # forward propagation
#         l0 = x
#
#
#     # matrix multiplication of input and hidden layer
#     l1 = sg.sigmoid(np.dot(l0, syn0))
#     # output layer
#     l2 = sg.sigmoid(np.dot(l1, syn1))
#     return l2
