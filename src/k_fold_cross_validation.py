k = 10
max_epochs = 130
stratified = False
structure_id = 0

from cross_validation import cross_validation, get_cv_tweets, test_proportion
from test_model import common_metrics
import numpy as np
from load import classes, model_name
from sklearn.utils import shuffle

if __name__ == "__main__":
	tweets, lengths = get_cv_tweets(classes)

	confusion_matrices = []
	expanded_confusion_matrices = []
	metrics = []

	train_tweets = []
	train_labels = []
	validate_tweets = []
	validate_labels = []
	test_tweets = []
	if stratified:
		for i in range(len(classes)):
			train = []
			validate = []
			test = []
			for j in range(len(classes[i])):
				tweets[i][j] = shuffle(tweets[i][j])
				train += tweets[i][j][:round(len(tweets[i][j]) * 0.8 * (1 - test_proportion))]
				validate += tweets[i][j][round(len(tweets[i][j]) * 0.8 * (1 - test_proportion)) : round(len(tweets[i][j]) * (1 - test_proportion))]
				test.append(tweets[i][j][round(len(tweets[i][j]) * (1 - test_proportion)):])
			train_tweets += train
			train_labels += [i] * len(train)
			validate_tweets += validate
			validate_labels += [i] * len(validate)
			test_tweets.append(test)
		train_tweets, train_labels = shuffle(train_tweets, train_labels)
		validate_tweets, validate_labels = shuffle(validate_tweets, validate_labels)
	else:
		all_tweets = []
		all_labels = []
		all_subclasses = []
		for i in range(len(classes)):
			count = 0
			test_tweets.append([])
			for j in range(len(classes[i])):
				test_tweets[i].append([])
				all_tweets += tweets[i][j]
				count += len(tweets[i][j])
				all_subclasses += [j] * len(tweets[i][j])
			all_labels += [i] * count
		all_tweets, all_labels, all_subclasses = shuffle(all_tweets, all_labels, all_subclasses)
		train_tweets = all_tweets[:round(len(all_tweets) * 0.8 * (1 - test_proportion))]
		train_labels = all_labels[:round(len(all_tweets) * 0.8 * (1 - test_proportion))]
		validate_tweets = all_tweets[round(len(all_tweets) * 0.8 * (1 - test_proportion)) : round(len(all_tweets) * (1 - test_proportion))]
		validate_labels = all_labels[round(len(all_tweets) * 0.8 * (1 - test_proportion)) : round(len(all_tweets) * (1 - test_proportion))]
		test = all_tweets[:round(len(all_tweets) * (1 - test_proportion))]
		test_labels = all_labels[:round(len(all_tweets) * (1 - test_proportion))]
		test_subclasses = all_subclasses[:round(len(all_tweets) * (1 - test_proportion))]
		for i in range(len(test)):
			test_tweets[test_labels[i]][test_subclasses[i]].append(test[i])

	confusion_matrix, expanded_confusion_matrix = cross_validation(0, model_name, max_epochs, train_tweets, train_labels, validate_tweets, validate_labels, test_tweets)
	confusion_matrices.append(confusion_matrix)
	expanded_confusion_matrices.append(expanded_confusion_matrix)
	metrics.append(common_metrics(confusion_matrix, expanded_confusion_matrix))
	print()
	print()

	confusion_matrix = np.sum(confusion_matrices, axis=0)
	expanded_confusion_matrix = np.sum(expanded_confusion_matrices, axis=0)
	metric = np.average(metrics, axis=0)
	print(confusion_matrix)
	print(expanded_confusion_matrix)
	print("Micro-averaged F1-score:", metric[0])
	print("Macro-averaged F1-score:", metric[1])
	print("Accuracy:", metric[2])

