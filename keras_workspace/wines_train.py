#Adapted from https://www.datacamp.com
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #set TensorFlow environment variable to silence INFO logs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read in white wine data
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')

# Read in red wine data
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')

#print(white.info())
#print(red.info())

# Last rows of `white`
#print(white.tail())

# Take a sample of 5 rows of `red`
#print(red.sample(5))

# Describe `white` category
#print(white.describe())

# Check for null values in `red`
#print(pd.isnull(red))

'''DATA VISUALIZATION'''
'''----Alcohol Bar Graphs---'''
fig, af = plt.subplots(1, 2)

af[0].hist(red.alcohol, 10, facecolor='red', alpha=0.5, label="Red wine")
af[1].hist(white.alcohol, 10, facecolor='white', ec="black", lw=0.5, alpha=0.5, label="White wine")

fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)
af[0].set_ylim([0, 1000])
af[0].set_xlabel("Alcohol in % Vol")
af[0].set_ylabel("Frequency")
af[1].set_xlabel("Alcohol in % Vol")
af[1].set_ylabel("Frequency")
af[0].legend(loc='best')
af[1].legend(loc='best')
fig.suptitle("Distribution of Alcohol in % Vol")

plt.show()

'''----Alcohol Histogram---'''
# print(np.histogram(red.alcohol, bins=[7,8,9,10,11,12,13,14,15]))
# print(np.histogram(white.alcohol, bins=[7,8,9,10,11,12,13,14,15]))

'''----Sulphates Bar Graphs---'''

fig, aq = plt.subplots(1, 2, figsize=(8, 4))

aq[0].scatter(red['quality'], red["sulphates"], color="red")
aq[1].scatter(white['quality'], white['sulphates'], color="white", edgecolors="black", lw=0.5)

aq[0].set_title("Red Wine")
aq[1].set_title("White Wine")
aq[0].set_xlabel("Wine Quality")
aq[1].set_xlabel("Wine Quality")
aq[0].set_ylabel("Sulphates grams/Litre")
aq[1].set_ylabel("Sulphates grams/Litre")
aq[0].set_xlim([0,10])
aq[1].set_xlim([0,10])
aq[0].set_ylim([0,2.5])
aq[1].set_ylim([0,2.5])
fig.subplots_adjust(wspace=0.5)
fig.suptitle("Wine Quality by Amount of Sulphates")

plt.show()


'''Volatile Acidity '''
np.random.seed(570)

redlabels = np.unique(red['quality'])
whitelabels = np.unique(white['quality'])

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
redcolors = np.random.rand(6, 4)
whitecolors = np.append(redcolors, np.random.rand(1, 4), axis=0)

for i in range(len(redcolors)):
    redy = red['alcohol'][red.quality == redlabels[i]]
    redx = red['volatile acidity'][red.quality == redlabels[i]]
    ax[0].scatter(redx, redy, c=redcolors[i])
for i in range(len(whitecolors)):
    whitey = white['alcohol'][white.quality == whitelabels[i]]
    whitex = white['volatile acidity'][white.quality == whitelabels[i]]
    ax[1].scatter(whitex, whitey, c=whitecolors[i])

ax[0].set_title("Red Wine")
ax[1].set_title("White Wine")
ax[0].set_xlim([0, 1.7])
ax[1].set_xlim([0, 1.7])
ax[0].set_ylim([5, 15.5])
ax[1].set_ylim([5, 15.5])
ax[0].set_xlabel("Volatile Acidity")
ax[0].set_ylabel("% Alcohol")
ax[1].set_xlabel("Volatile Acidity")
ax[1].set_ylabel("% Alcohol")
ax[0].legend(redlabels, loc='best', bbox_to_anchor=(1.3, 1))
ax[1].legend(whitelabels, loc='best', bbox_to_anchor=(1.3, 1))
fig.suptitle("Alcohol - Volatile Acidity")
fig.subplots_adjust(top=0.85, wspace=0.7)

plt.show()

'''Preprocess Data for Feature Correlation'''

white['type'] = 0
red['type'] = 1

wines = red.append(white, ignore_index=True)

import seaborn as sns

correlation = wines.corr()
sns.plt.title("Wine Features Correlation")
sns.heatmap(correlation, xticklabels=correlation.columns.values, yticklabels=correlation.columns.values)

_, xlabels = plt.xticks()
plt.setp(xlabels, rotation=90)

_, ylabels = plt.yticks()
plt.setp(ylabels, rotation=0)
plt.show()

#leave last column 'type'
X = wines.ix[:,0:11]

#Specify target label and flatten the array
# type 0 = white wine
# type 1 = red wine
y = np.ravel(wines.type)

from sklearn.model_selection import train_test_split

#set a third of data as test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.preprocessing import StandardScaler

#define scaler
scaler = StandardScaler().fit(X_train)

#scale the training set
X_train = scaler.transform(X_train)

#scale the testing set
X_test = scaler.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense

#instantiate neural network model (linear stack)
model = Sequential()

# Add an input layer
#add 12 feature data as input. Use Relu function as activation function
#Dense means a fully connected network
#relu activation function good against diminishing gradients
#good read on vanishing gradients here: http://neuralnetworksanddeeplearning.com/chap5.html

#Dense layer performs: output = activation(dot(input, kernel) + bias)
# The model will take as input arrays of shape (*, 11)
# and output arrays of shape (*, 12)
model.add(Dense(12, activation='relu', input_shape=(11,)))

# Add second hidden layer with 8 nodes, also uses relu
model.add(Dense(8, activation='relu'))

# Add an output layer with one node, use sigmoid activation
model.add(Dense(1, activation='sigmoid'))

print("Output Shape:", model.output_shape)
print("Model Summary:", model.summary())
print("Model Config:", model.get_config())
print("Model Weights:", model.get_weights())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=1)

y_pred = model.predict(X_test)


score = model.evaluate(X_test, y_test, verbose=1)
print("Keras Model Score:", score)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

# Confusion matrix. Breakdown of predictions into a table showing correct predictions and the types of incorrect predictions made
print("y_test:", y_test[:5])
print("y_pred", y_pred[:5].round())

print("Sklearn Confusion Matrix:", confusion_matrix(y_test, y_pred.round()))

# Precision. A measure of a classifier’s exactness.
#precision_score("Sklearn Precision Score:", y_test, y_pred.round())

# Recall. A measure of a classifier’s completeness. The higher the recall, the more cases the classifier covers.
#print("Sklearn Recall:", recall_score(y_test, y_pred.round()))

# F1 score. Weighted average of precision and recall.
#print("Sklearn F1 Score:", f1_score(y_test,y_pred.round()))

# Cohen's kappa. Classification accuracy normalized by the imbalance of the classes in the data.
#print("Sklearn Cohen's Kappa:", cohen_kappa_score(y_test, y_pred.round()))