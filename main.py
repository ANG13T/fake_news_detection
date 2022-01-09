import pickle
import numpy as np
from os import path

from bs4 import BeautifulSoup as bs
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

if not path.exists('final_data.pkl'):
	print('no saved data was found; generating from scratch...')
	print('loading data')
	# structure of each item: url, html, (1 if fake else 0)
	with open('train_val_data.pkl', 'rb') as f:
		train_data, val_data = pickle.load(f)
	with open('test_data.pkl', 'rb') as f:
		test_data = pickle.load(f)

	print('making Tokenizer')
	tokenizer = Tokenizer(
		num_words=12_000,  # TUNABLE
		filters='!"#$%&()*+,-./…‘’“”—–:;<=>?@[\\]^_`{|}~\t\n©®™',
		lower=True,
		split=" "
	)

	train_data.pop(232)  # for some reason they cause the parser to hang
	train_data.pop(301)
	train_data.pop(620)
	train_data.pop(1362)
	train_data.pop(1656)
	train_data.pop(1738)

	if not path.exists('text_data.pkl'):
		print('no saved text found; converting HTML to text')
		train_texts = [bs(page[1], 'html.parser').get_text() for page in train_data]
		valid_texts = [bs(page[1], 'html.parser').get_text() for page in val_data]
		test_texts = [bs(page[1], 'html.parser').get_text() for page in test_data]

		with open('text_data.pkl', 'wb') as f:
			pickle.dump((train_texts, valid_texts, test_texts), f)
	else:
		print('using preconverted text')
		with open('text_data.pkl', 'rb') as f:
			train_texts, valid_texts, test_texts = pickle.load(f)

	print('fitting Tokenizer')
	tokenizer.fit_on_texts(train_texts)
	total_words = len(tokenizer.word_index)

	print('generating sequences and labels from data/text from earlier')
	X_train = tokenizer.texts_to_sequences(train_texts)
	X_valid = tokenizer.texts_to_sequences(valid_texts)
	X_test = tokenizer.texts_to_sequences(test_texts)
	y_train = [page[2] for page in train_data]
	y_valid = [page[2] for page in val_data]
	y_test = [page[2] for page in test_data]

	print('pruning bad data')

	to_pop = []
	for i in range(len(X_train)):
		content = train_texts[i]
		sequence = X_train[i]
		if len(sequence) < 15:
			to_pop.append(i)
		elif len(sequence) < 30 and ('403' in content or '404' in content or '401' in content or '500' in content or '502' in content or '503' in content):
			to_pop.append(i)
	for offset, idx_to_pop in enumerate(to_pop):
		X_train.pop(idx_to_pop - offset)  # the array shrinks when we pop, so account for that. This only works since we know the indexes are sorted low-to-high.
		y_train.pop(idx_to_pop - offset)
		# no need to pop the texts since they're deleted
	del to_pop, train_texts
	to_pop = []
	for i in range(len(X_valid)):
		content = valid_texts[i]
		sequence = X_valid[i]
		if len(sequence) < 15:
			to_pop.append(i)
		elif len(sequence) < 30 and ('403' in content or '404' in content or '401' in content or '500' in content or '502' in content or '503' in content):
			to_pop.append(i)
	for offset, idx_to_pop in enumerate(to_pop):
		X_valid.pop(idx_to_pop - offset)
		y_valid.pop(idx_to_pop - offset)
	del to_pop, valid_texts
	to_pop = []
	for i in range(len(X_test)):
		content = test_texts[i]
		sequence = X_test[i]
		if len(sequence) < 15:
			to_pop.append(i)
		elif len(sequence) < 30 and ('403' in content or '404' in content or '401' in content or '500' in content or '502' in content or '503' in content):
			to_pop.append(i)
	for offset, idx_to_pop in enumerate(to_pop):
		X_test.pop(idx_to_pop - offset)
		y_test.pop(idx_to_pop - offset)
	del to_pop, test_texts

	word_idx = tokenizer.word_index
	breakpoint()
	del val_data, tokenizer, train_data
	with open('final_data.pkl', 'wb') as f:
		pickle.dump((X_train, y_train, X_valid, y_valid, X_test, y_test, total_words, word_idx), f)
else:
	print('using saved data')
	with open('final_data.pkl', 'rb') as f:
		X_train, y_train, X_valid, y_valid, X_test, y_test, total_words, word_idx = pickle.load(f)
	del X_test, y_test

# the word numbers start at 1 but we want the one-hot encoding array to start at zero.
# example with 10 words: 6 -> 0 0 0 0 0 1 0 0 0 0
# there are five elements before the 1 (word_as_num - 1) and four after (total_words - word_as_num)
# one_hot_encoded_sequences = [[np.concatenate((np.zeros(word_as_num - 1, dtype=np.uint8), np.array([1], dtype=np.uint8), np.zeros(total_words - word_as_num, dtype=np.uint8))) for word_as_num in sequence] for sequence in sequences]
# ^ commented because it isn't needed by Embedding but might be good to have in the future.

### LOAD THE EMBEDDINGS

if not path.exists('embedding.pkl'):
	# Load in embeddings
	glove_vectors = 'glove.6B.100d.txt'
	print('making embeddings')
	glove = np.loadtxt(glove_vectors, dtype='str', comments=None)

	# Extract the vectors and words
	vectors = glove[:, 1:].astype('float')
	words = glove[:, 0]

	del glove

	# Create lookup of words to vectors
	word_lookup = {word: vector for word, vector in zip(words, vectors)}

	# New matrix to hold word embeddings
	embedding_matrix = np.zeros((total_words, vectors.shape[1]))

	for i, word in enumerate(word_idx.keys()):
		# Look up the word embedding
		vector = word_lookup.get(word, None)

		# Record in matrix
		if vector is not None:
			embedding_matrix[i, :] = vector

	del word_lookup, vectors, words
	with open('embedding.pkl', 'wb') as f:
		pickle.dump(embedding_matrix, f)
else:
	print('using premade embeddings')
	with open('embedding.pkl', 'rb') as f:
		embedding_matrix = pickle.load(f)

### MAKE THE MODEL

model = Sequential()

# Embedding layer, to convert the words to embeddings
model.add(Embedding(
	input_dim=total_words,
	output_dim=100,  # TUNABLE
	mask_zero=True,  # since 0 means word not in vocabulary, we mask it.
	weights=[embedding_matrix],  # use the pre-trained embeddings
	trainable=False,  # don't train this since it's pre-trained
))

# Masking layer for pre-trained embeddings, since the words in the pre-trained embeddings aren't guaranteed to match those in our Tokenizer's encoding
model.add(Masking(mask_value=0.0))

# Recurrent layer, to actually do the RNN stuff
model.add(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1, input_shape=(None, 100)))  # input shape: number of timesteps, dimensions in each timestep (100 from the embedding)

# Fully connected layer, "adds additional representational capacity to the network"
model.add(Dense(64, activation='relu'))

# Dropout to prevent overfitting
model.add(Dropout(0.5))

# Output layer using softmax to make a probability
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

### TRAIN THE MODEL!
callbacks = [
	EarlyStopping(monitor='val_loss', patience=5),  # stop early if we're overfitting
	ModelCheckpoint('model.h5', save_best_only=True, save_weights_only=False)
]

print('fitting model')

history = model.fit(pad_sequences(X_train), to_categorical(y_train), batch_size=10, epochs=150, callbacks=callbacks, validation_data=(pad_sequences(X_valid), to_categorical(y_valid)))
