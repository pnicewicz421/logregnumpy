import numpy as np
import pandas as pd
import os

# 1) load the data 
dir = '/Users/peternicewicz/documents/data science/mnist/'
os.chdir(dir)

data = pd.read_csv('train.csv', sep=',')

# separate x from y (target)
labels = data.loc[:, 'label']
features = data.drop(labels='label', axis=1)

# data is already flattened. normalize
features = features / 255

#labels = labels.reshape((4200, 1))

label_column = np.zeros((42000, 10))

# make labels categorical
for n in xrange(len(labels)):
	lbl = labels[n]
	label_column[n, lbl] = 1 

labels = label_column



loss_hist = []
lr = 10**-4

epochs = 1000

# define activation function
def sigmoid(z):
	return 1 / (1 + np.e**-z)

# initalize weights randomly
def initialize_weights(features):
	# weights will be initialized in the range [-1, 1]
	weights = 2 * np.random.rand(np.shape(features)[1], np.shape(labels)[1]) - 1 
	return weights

def predict(features, weights):
	return np.dot(features, weights)

def calculate_loss(features=features, labels=labels):
	N = np.shape(features)[0] * np.shape(features)[1]
	predictions = sigmoid(predict(features, weights))
	#when label=1
	class1_loss = -labels*np.log(predictions)
	class2_loss = (1-labels)*np.log(1-predictions)

	loss = class1_loss + class2_loss
	loss = loss.sum() / N
	return loss

def update_weights(features=features, targets=labels, weights=weights, lr=lr):
	# set up weight derivate matrix
	predictions = predict(features, weights)
	error = targets - predictions
	gradient = np.dot(-features.T, error)
	gradient /= np.shape(features)[0]
	gradient *= lr
	weights -= gradient
	return weights

# train procedure
weights = initialize_weights(features)

for e in xrange(epochs):
	# measure the loss function
	weights = update_weights()
	loss = calculate_loss()

	loss_hist.append(loss)
	if e % 10 == 0:
		print "Epoch: " + str(e) + ", Cost: " + str(loss)

