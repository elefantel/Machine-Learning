import tensorflow as tf
import pandas as pd
import numpy as np
'''Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. 
Irvine, CA: University of California, School of Information and Computer Science.'''

# Read in white wine data
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
# Read in red wine data
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')

redlabels = np.unique(red['quality'])
whitelabels = np.unique(white['quality'])

x = tf.constant(1.0, name='input')
w = tf.Variable(0.8, name='weight')
y = tf.multiply(w, x, name='output')
y_ = tf.constant(0.0, name='correct_value')
loss = tf.pow(y - y_, 2, name='loss')
train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)

for value in [x, w, y, y_, loss]:
    tf.summary.scalar(value.op.name, value)

summaries = tf.summary.merge_all()

sess = tf.Session()
file_writer = tf.summary.FileWriter('/Users/Mpendulo/PycharmProjects/Machine-Learning/tensorflow_workspace', sess.graph)

sess.run(tf.global_variables_initializer())
for i in range(100):
    file_writer.add_summary(sess.run(summaries), i)
    sess.run(train_step)

