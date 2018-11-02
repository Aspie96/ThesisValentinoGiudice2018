iterations = 1
max_epochs = 120
test_proportion = 0.1
stratified = True
multiply_classes = True

import numpy as np
from test_model import test_model, common_metrics
from load import add_tweets, classes, binary, model_name, data_folder, subclasses
import codecs
from sklearn.utils import shuffle
from create_model import create_model

def cross_validation(structure_id, model_name, max_epochs, train_tweets, train_labels, validate_tweets, validate_labels, test_tweets):
	train_tweetsA = []
	train_labelsA = []
	validate_tweetsA = []
	validate_labelsA = []
	for i in range(len(train_tweets)):
		if train_labels[i] == 0:
			train_tweetsA.append(train_tweets[i])
			train_labelsA.append(train_labels[i])
		elif train_labels[i] == 1:
			train_tweetsA.append(train_tweets[i])
			train_labelsA.append(train_labels[i])
		else:
			train_tweetsA.append(train_tweets[i])
			train_labelsA.append(train_labels[i])
			train_labelsA[i] = 1
	for i in range(len(validate_tweets)):
		if validate_labels[i] == 0:
			validate_tweetsA.append(validate_tweets[i])
			validate_labelsA.append(validate_labels[i])
		elif validate_labels[i] == 1:
			validate_tweetsA.append(validate_tweets[i])
			validate_labelsA.append(validate_labels[i])
		else:
			validate_tweetsA.append(validate_tweets[i])
			validate_labelsA.append(validate_labels[i])
			validate_labelsA[i] = 1
	train_tweetsB = []
	train_labelsB = []
	validate_tweetsB = []
	validate_labelsB = []
	for i in range(len(train_tweets)):
		if train_labels[i] == 1:
			train_tweetsB.append(train_tweets[i])
			train_labelsB.append(train_labels[i] - 1)
		elif train_labels[i] == 2:
			train_tweetsB.append(train_tweets[i])
			train_labelsB.append(train_labels[i] - 1)
	for i in range(len(validate_tweets)):
		if validate_labels[i] == 1:
			validate_tweetsB.append(validate_tweets[i])
			validate_labelsB.append(validate_labels[i] - 1)
		elif validate_labels[i] == 2:
			validate_tweetsB.append(validate_tweets[i])
			validate_labelsB.append(validate_labels[i] - 1)
	create_model(0, model_name, classes, max_epochs, train_tweets, train_labels, validate_tweets, validate_labels)
	return test_model(model_name, classes, subclasses, binary, test_tweets)


def get_cv_tweets(classes):
	tweets = []
	lengths = []
	for i in range(len(classes)):
		a = []
		b = []
		length = 0
		for j in range(len(classes[i])):
			t = []
			fp = codecs.open(data_folder + "/" + classes[i][j] + "_cv.txt", "r", "utf-8")
			ts = fp.readlines()
			add_tweets(ts, t)
			fp.close()
			a.append(t)
			b.append(ts)
			length += len(t)
		tweets.append(a)
		lengths.append(length)
	return tweets, lengths


if __name__ == "__main__":
	confusion_matrices = []
	expanded_confusion_matrices = []
	tweets, lengths = get_cv_tweets(classes)
	metrics = []

	for i in range(iterations):
		train_tweets = []
		train_labels = []
		validate_tweets = []
		validate_labels = []
		test_tweets = []
		test_labels = []
		if stratified:
			for j in range(len(classes)):
				train = []
				validate = []
				test = []
				for k in range(len(classes[j])):
					tweets[j][k] = shuffle(tweets[j][k])
					train += tweets[j][k][:round(len(tweets[j][k]) * 0.8 * (1 - test_proportion))]
					validate += tweets[j][k][round(len(tweets[j][k]) * 0.8 * (1 - test_proportion)) : round(len(tweets[j][k]) * (1 - test_proportion))]
					test.append(tweets[j][k][round(len(tweets[j][k]) * (1 - test_proportion)):])
				train_tweets += train
				train_labels += [j] * len(train)
				validate_tweets += validate
				validate_labels += [j] * len(validate)
				test_tweets.append(test)
			train_tweets, train_labels = shuffle(train_tweets, train_labels)
			validate_tweets, validate_labels = shuffle(validate_tweets, validate_labels)
		else:
			all_tweets = []
			all_labels = []
			all_subclasses = []
			for j in range(len(classes)):
				count = 0
				test_tweets.append([])
				for k in range(len(classes[j])):
					test_tweets[j].append([])
					all_tweets += tweets[j][k]
					count += len(tweets[j][k])
					all_subclasses += [j] * len(tweets[j][k])
				all_labels += [j] * count
			all_tweets, all_labels, all_subclasses = shuffle(all_tweets, all_labels, all_subclasses)
			train_tweets = all_tweets[:round(len(all_tweets) * 0.8 * (1 - test_proportion))]
			train_labels = all_labels[:round(len(all_tweets) * 0.8 * (1 - test_proportion))]
			validate_tweets = all_tweets[round(len(all_tweets) * 0.8 * (1 - test_proportion)) : round(len(all_tweets) * (1 - test_proportion))]
			validate_labels = all_labels[round(len(all_tweets) * 0.8 * (1 - test_proportion)) : round(len(all_tweets) * (1 - test_proportion))]
			test = all_tweets[:round(len(all_tweets) * (1 - test_proportion))]
			test_labels = all_labels[:round(len(all_tweets) * (1 - test_proportion))]
			test_subclasses = all_subclasses[:round(len(all_tweets) * (1 - test_proportion))]
			for j in range(len(test)):
				test_tweets[test_labels[j]][test_subclasses[j]].append(test[j])
		
		confusion_matrix, expanded_confusion_matrix = cross_validation(0, model_name, max_epochs, train_tweets, train_labels, validate_tweets, validate_labels, test_tweets)
		confusion_matrices.append(confusion_matrix)
		expanded_confusion_matrices.append(expanded_confusion_matrix)
		metrics.append(common_metrics(confusion_matrix, expanded_confusion_matrix))
	print()
	print()
	for i in range(iterations):
		print(confusion_matrices[i])
		print(expanded_confusion_matrices[i])
		print(metrics[i])
		print()
	confusion_matrix = np.sum(confusion_matrices, axis=0)
	expanded_confusion_matrix = np.sum(expanded_confusion_matrices, axis=0)
	metric = np.average(metrics, axis=0)
	print(confusion_matrix)
	print(expanded_confusion_matrix)
	print("Micro-averaged F1-score:", metric[0])
	print("Macro-averaged F1-score:", metric[1])
	print("Accuracy:", metric[2])
