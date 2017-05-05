#natural language toolkit

import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime

stemmer = LancasterStemmer()

# create 3 classes of training data
training = []
training.append({"class":"greeting", "sentence":"how are you?"})
training.append({"class":"greeting", "sentence":"how is your day?"})
training.append({"class":"greeting", "sentence":"good day"})
training.append({"class":"greeting", "sentence":"good morning"})
training.append({"class":"greeting", "sentence":"how is it going today?"})

training.append({"class":"goodbye", "sentence":"have a nice day"})
training.append({"class":"goodbye", "sentence":"see you later"})
training.append({"class":"goodbye", "sentence":"have a nice day"})
training.append({"class":"goodbye", "sentence":"talk to you soon"})
training.append({"class":"goodbye", "sentence":"till we meet again"})

training.append({"class":"command", "sentence":"print me a report"})
training.append({"class":"command", "sentence":"can you compile a report?"})
training.append({"class":"command", "sentence":"having a difficult time compiling the report?"})
training.append({"class":"command", "sentence":"what's the review for?"})
training.append({"class":"command", "sentence":"reviewing your email now"})
print ("%s sentences in training data" % len(training))

words = []
classes = []
documents = []
ignore_words = ['?']

# iterate over each sentence in training set
for pattern in training:
    # tokenize each word in the sentence

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
print (len(words), "unique stemmed words", words)