import os 

dir_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/" + os.pardir)

task = 0

all_classes = [
	[
		[
			"subclass1",
			"subclass2"
		],
		[
			"subclass3",
			"subclass4"
		]
	]
]

classes = all_classes[task]

subclasses = []
for cl in classes:
	subclasses += cl

all_data_folders = [
	"data1"
]

all_data_folders = [dir_path + "/data/" + data_folder for data_folder in all_data_folders]

data_folder = all_data_folders[task]

binary = len(classes) == 2

model_name = "model1.h5"

import re
import numpy as np
from emoji import unicode_codes

dictionary = [" ", "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "=", "?", "@", "[", "]", "_", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "|", "~"]

input_len = len(dictionary) + 6
def add_tweets(tweets, output):
	""" Adds the given tweets to the given array, representing it in a way which can be used by the model.
	@param  tweets  The tweets to be added.
	@param  output  The array to be inserted the resulting representations into.
	"""
	for tweet in tweets:
		tweet = tweet.replace("\n", "")
		tweet = re.sub(r" +", " ", tweet)
		tweet = re.sub(r"^ ", "", tweet)
		tweet = re.sub(r" $", "", tweet)
		i = 0
		sequence = []
		start_of_emoji = False
		emoji_length = 0
		while i < len(tweet) and len(sequence) < 140:
			c = tweet[i]
			array = [0] * input_len
			if emoji_length > 0:
				if start_of_emoji:
					array[-4] = 1
					start_of_emoji = False
				array[-3] = 1
				emoji_length -= 1
			if c == "\u00E0" or c == "\u00E1":
				array[-2] = 1
				c = "a"
			elif c == "\u00E8" or c == "\u00E9":
				array[-2] = 1
				c = "e"
			elif c == "\u00EC" or c == "\u00ED":
				array[-2] = 1
				c = "i"
			elif c == "\u00F2" or c == "\u00F3":
				array[-2] = 1
				c = "o"
			elif c == "\u00F9" or c == "\u00FA":
				array[-2] = 1
				c = "u"
			elif c == "\u00C0" or c == "\u00C1":
				array[-2] = 1
				c = "A"
			elif c == "\u00C8" or c == "\u00C9":
				array[-2] = 1
				c = "E"
			elif c == "\u00CC" or c == "\u00CD":
				array[-2] = 1
				c = "I"
			elif c == "\u00D2" or c == "\u00D3":
				array[-2] = 1
				c = "O"
			elif c == "\u00D9" or c == "\u00DA":
				array[-2] = 1
				c = "U"
			if "A" <= c <= "Z":
				c = c.lower()
				array[-1] = 1
			if not c in dictionary and c in unicode_codes.UNICODE_EMOJI:
				start_of_emoji = True
				code = unicode_codes.UNICODE_EMOJI.get(c).replace(":", "")
				tweet = tweet[: i] + code + tweet[i + 1:]
				emoji_length= len(code)
			else:
				if c in dictionary:
					if "a" <= c <= "z":
						array[-5] = 1
					elif "0" <= c <= "9":
						array[-6] = 1
					array[dictionary.index(c)] = 1
					sequence += [array]
				i += 1
		padding = [0] * input_len
		sequence = [padding] * (140 - len(sequence)) + sequence
		output.append(np.array(sequence))

print("task:", task)
print("classes:", classes)
print("binary:", binary)
print("model_name:", model_name)
print()
