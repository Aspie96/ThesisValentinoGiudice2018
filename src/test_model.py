from keras.models import load_model
import numpy as np
from keras import backend as K
from load import add_tweets, classes, binary, model_name, data_folder, subclasses
from sklearn.utils import shuffle
import codecs

def test_model(model_name, classes, subclasses, binary, test_tweets):
	def f1(y_true, y_pred):
		def recall(y_true, y_pred):
			true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
			possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
			recall = true_positives / (possible_positives + K.epsilon())
			return recall
		def precision(y_true, y_pred):
			true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
			predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
			precision = true_positives / (predicted_positives + K.epsilon())
			return precision
		precision = precision(y_true, y_pred)
		recall = recall(y_true, y_pred)
		return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

	model = load_model("../models/" + model_name)

	confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)
	expanded_confusion_matrix = np.zeros((len(subclasses), len(classes)), dtype=int)
	subclass_counter = 0
	for i in range(len(test_tweets)):
		for j in range(len(test_tweets[i])):
			results = model.predict(np.array(test_tweets[i][j]))
			for k in range(len(test_tweets[i][j])):
				if binary:
					if results[k][0] == 0.5:
						predicted_as = 1 - i
					else:
						predicted_as = int(round(results[k][0]))
				else:
					predicted_as = int(round(results[k][0]))
				expanded_confusion_matrix[subclass_counter, predicted_as] += 1
				confusion_matrix[i, predicted_as] += 1
			subclass_counter += 1
	print(expanded_confusion_matrix)
	
	return confusion_matrix, expanded_confusion_matrix


def common_metrics(confusion_matrix, expanded_confusion_matrix):
	precisions = []
	recalls = []
	trueps = []
	falsens = []
	falseps = []
	for i in range(len(classes)):
		truep = confusion_matrix[i, i]
		falsen = 0
		falsep = 0
		for j in range(len(classes)):
			if j != i:
				falsen += confusion_matrix[i, j]
				falsep += confusion_matrix[j, i]
		trueps.append(truep)
		falsens.append(falsen)
		falseps.append(falsep)
		if truep == 0:
			precision = 0
		else:
			precision = truep / (truep + falsep)
		recall = truep / (truep + falsen)
		precisions.append(precision)
		recalls.append(recall)
		if precision == 0 and recall == 0:
			f1 = 0
		else:
			f1 = 2 * precision * recall / (precision + recall)
		print("Class", i, "Precision:", precision, "Recall:", recall, "F1-score:", f1)
	micro_avg_precision = sum(trueps) / (sum(trueps) + sum(falseps))
	micro_avg_recall = sum(trueps) / (sum(trueps) + sum(falsens))
	print(micro_avg_precision, micro_avg_recall)
	if micro_avg_precision == 0 and micro_avg_recall == 0:
		micro_avg_f1score = 0
	else:
		micro_avg_f1score = 2 * micro_avg_precision * micro_avg_recall / (micro_avg_precision + micro_avg_recall)
	print("Micro-averaged F1-score:", micro_avg_f1score)
	macro_avg_precision = sum(precisions) / len(classes)
	macro_avg_recall = sum(recalls) / len(classes)
	if macro_avg_precision == 0 and macro_avg_recall == 0:
		macro_avg_f1score = 0
	else:
		macro_avg_f1score = 2 * macro_avg_precision * macro_avg_recall / (macro_avg_precision + macro_avg_recall)
	print("Macro-averaged F1-score:", macro_avg_f1score)
	accuracy = sum(trueps) / confusion_matrix.sum()
	print("Accuracy:", accuracy)
	return micro_avg_f1score, macro_avg_f1score, accuracy


if __name__ == "__main__":
	test_tweets = []

	tweets = []
	for i in range(len(classes)):
		a = []
		length = 0
		for j in range(len(classes[i])):
			t = []
			fp = codecs.open(data_folder + "/" + classes[i][j] + "_test.txt", "r", "utf-8")
			ts = fp.readlines()
			add_tweets(ts, t)
			fp.close()
			a.append(t)
			length += len(t)
		tweets.append(a)
	for i in range(len(classes)):
		test = []
		for j in range(len(classes[i])):
			test.append(tweets[i][j])
		test_tweets.append(test)
	confusion_matrix, expanded_confusion_matrix = test_model(model_name, classes, subclasses, binary, test_tweets)
	print()
	common_metrics(confusion_matrix, expanded_confusion_matrix)
