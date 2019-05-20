
# %%
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Input, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
import pandas as pd
import nltk
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# ## Data Exploration

# %%
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
test_labels = pd.read_csv("data/test_labels.csv")


# %%
len(train_data), len(test_data)


# %%
train_data.head()


# %%
train_data["id"].head()


# %%
train_data.isnull().values.any()


# %%
test_data.isnull().values.any()


# %%
train_data.isnull().any()


# %%
test_data.isnull().any()


# %%
classes = ["toxic", "severe_toxic", "obscene",
           "threat", "insult", "identity_hate"]
train_y = train_data[classes].values
train_x = train_data["comment_text"]
test_x = test_data["comment_text"]


# %%
train_x.head()


# %%
test_x.head()

# %% [markdown]
# ## Data Preprocessing

# %%
max_words = 20000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(list(train_x))


# %%
tokenized_train_x = tokenizer.texts_to_sequences(train_x)
tokenized_test_x = tokenizer.texts_to_sequences(test_x)


# %%
len(tokenized_train_x), len(tokenized_test_x)


# %%
maxlen = 200
X_train = pad_sequences(tokenized_train_x, maxlen=maxlen)
X_test = pad_sequences(tokenized_test_x, maxlen=maxlen)


# %%
len(X_train), len(X_test)


# %%
X_train[23]


# %%
total_num_words = [len(one_comment) for one_comment in tokenized_train_x]
# total_num_words


# %%
plt.hist(total_num_words, bins=np.arange(0, 410, 10))
plt.show()

# %% [markdown]
# ## Final Preprocessed Data

# %%
train_x[0]


# %%
X_train[0]

# %% [markdown]
# ### Stopwords

# %%
%load_ext autoreload
%autoreload 2
from utils.preprocessing import filter_stop_words
filtered_train_x = filter_stop_words(train_x)
filtered_test_x = filter_stop_words(test_x)

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(list(filtered_train_x))

# %%
tokenized_train_x = tokenizer.texts_to_sequences(filtered_train_x)
tokenized_test_x = tokenizer.texts_to_sequences(filtered_test_x)

# %%
total_num_words = [len(one_comment) for one_comment in tokenized_train_x]


# %% [markdown]
# Total number of words after removing stopwords.

# %%
plt.hist(total_num_words, bins=np.arange(0, 410, 10))
plt.show()

# %% [markdown]
# ### Lemmatization
%autoreload 2
from utils.preprocessing import lemmatize

lemmatized_train_x = lemmatize(filtered_train_x)
lemmatized_test_x = lemmatize(filtered_test_x)

tokenizer.fit_on_texts(list(lemmatized_train_x))

# %%
tokenized_train_x = tokenizer.texts_to_sequences(lemmatized_train_x)
tokenized_test_x = tokenizer.texts_to_sequences(lemmatized_test_x)
# %%
total_num_words = [len(one_comment) for one_comment in tokenized_train_x]

# %% [markdown]
# Total number of words after Lemmatization + Stop words correction.
# %%
plt.hist(total_num_words, bins=np.arange(0, 410, 10))
plt.show()

# %% [markdown]
# ### Stemming
%autoreload 2
from utils.preprocessing import stem

stemmed_train_x = stem(filtered_train_x)
stemmed_test_x = stem(filtered_test_x)

tokenizer.fit_on_texts(list(stemmed_train_x))

# %%
tokenized_train_x = tokenizer.texts_to_sequences(stemmed_train_x)
tokenized_test_x = tokenizer.texts_to_sequences(stemmed_test_x)
# %%
total_num_words = [len(one_comment) for one_comment in tokenized_train_x]

# %% [markdown]
# Total number of words after Stemming + Stop words correction.
# %%
plt.hist(total_num_words, bins=np.arange(0, 410, 10))
plt.show()

# %%
print(train_x[0], "\n")
print(filtered_train_x[0], "\n")
print(lemmatized_train_x[0], "\n")
print(stemmed_train_x[0], "\n")