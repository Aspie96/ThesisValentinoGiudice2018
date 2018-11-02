improve = False
balanced = True
max_epochs = 40
tollerance = 8

from sklearn.utils import shuffle
from keras import Sequential
from keras.layers import Dropout, GRU, Dense, Bidirectional, Conv1D, GaussianNoise
import numpy as np
from keras.utils import to_categorical
from sklearn.utils import class_weight
from load import input_len, add_tweets, classes, binary, model_name, data_folder
import codecs
from math import inf

def get_weight_multipliers(predictions, labels):
	confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)
	for i in range(len(predictions)):
		if binary:
			predicted_as = int(round(predictions[i][0]))
		else:
			predicted_as = predictions[i].argmax()
		actual = labels[i]
		confusion_matrix[predicted_as, actual] += 1
	weight_multipliers = [0] * len(classes)
	for i in range(len(classes)):
		truep = confusion_matrix[i, i]
		falsep = confusion_matrix[i].sum() - truep
		falsen = confusion_matrix[:,i].sum() - truep
		weight_multipliers[i] = (falsen + truep + 1) / (falsep + truep + 1)
	print(confusion_matrix)
	print(weight_multipliers)
	return weight_multipliers

def create_model(structure_id, model_name, classes, max_epochs, train_tweets, train_labels, validate_tweets, validate_labels):
	model = Sequential()
	if structure_id == 0:
		binary = True
		model.add(GaussianNoise(0.1, input_shape=(140, input_len)))
		model.add(Conv1D(8, 3, activation="relu"))
		model.add(Dropout(0.3))
		model.add(Conv1D(8, 3, activation="relu"))
		model.add(Dropout(0.3))
		model.add(Conv1D(8, 3, activation="relu"))
		model.add(Dropout(0.3))
		model.add(Bidirectional(GRU(6, recurrent_dropout=0.5)))
		model.add(Dropout(0.3))
	if binary:
		model.add(Dense(1, activation="sigmoid"))
		model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
	else:
		model.add(Dense(len(classes), activation="softmax"))
		model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

	losses = []
	if binary:
		train_outputs = train_labels
		validate_outputs = validate_labels
	else:
		train_outputs = to_categorical(train_labels)
		validate_outputs = to_categorical(validate_labels)

	bestLoss = inf;
	step = 0
	i = 0
	if balanced:
		train_weights = class_weight.compute_class_weight("balanced", np.unique(train_labels), train_labels)
		validate_weights = class_weight.compute_class_weight("balanced", np.unique(validate_labels), validate_labels)
		sample_weights = []
		print(train_weights)
		print(validate_weights)
		for lable in validate_labels:
			sample_weights.append(validate_weights[lable])
		sample_weights = np.array(sample_weights)
	else:
		train_weights = None
	got_worse = 0
	best_train_weights = train_weights.copy()
	while step < max_epochs and i < 130 and got_worse < tollerance:
		i += 1
		step += 0
		train_sample_weights = []
		print(train_weights)
		print(validate_weights)
		for lable in train_labels:
			train_sample_weights.append(best_train_weights[lable])
		train_sample_weights = np.array(train_sample_weights)

		print(best_train_weights)
		model.fit(np.array(train_tweets), np.array(train_outputs), epochs=1, batch_size=16, sample_weight=train_sample_weights)
		loss = model.evaluate(np.array(validate_tweets), np.array(validate_outputs), sample_weight=sample_weights)[0]
		losses += [loss]
		if loss > bestLoss:
			got_worse += 1
		else:
			if loss < bestLoss:
				model.save("../models/" + model_name)
				bestLoss = loss
			got_worse = 0
		print(step, loss, bestLoss)
		predictions = model.predict(np.array(validate_tweets))
		get_weight_multipliers(predictions, validate_labels)

if __name__ == "__main__":
	train_tweets = []
	train_labels = []
	for i in range(len(classes)):
		fp = codecs.open("../data/" + data_folder + "/" + classes[i] + "_train.txt", "r", "utf-8")
		tweets = fp.readlines()
		add_tweets(tweets, train_tweets)
		fp.close()
		train_labels += [i] * len(tweets)
	train_tweets, train_labels = shuffle(train_tweets, train_labels)

	validate_tweets = []
	validate_labels = []
	for i in range(len(classes)):
		fp = codecs.open("../data/" + classes[i] + "_validate.txt", "r", "utf-8")
		tweets = fp.readlines()
		add_tweets(tweets, validate_tweets)
		fp.close()
		validate_labels += [i] * len(tweets)
	create_model(model_name, improve, False, max_epochs, train_tweets, train_labels, validate_tweets, validate_labels)
