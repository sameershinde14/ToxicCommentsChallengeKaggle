
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

from utils.data_loader import load_train_data, load_test_data

# %% [markdown]
# ## Data Exploration

# %%
train_data, valid_data = load_train_data("data/train.csv")
loaded_test_data = load_test_data("data/test.csv", "data/test_labels.csv")
test_data = loaded_test_data[["id", "comment_text"]]
test_labels = loaded_test_data[
    [
        "toxic", "severe_toxic", "obscene",
        "threat", "insult", "identity_hate"
    ]
]

# %%
len(train_data), len(valid_data), len(test_data)


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
valid_y = valid_data[classes].values
valid_x = valid_data["comment_text"]
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
tokenized_valid_x = tokenizer.texts_to_sequences(valid_x)
tokenized_test_x = tokenizer.texts_to_sequences(test_x)


# %%
len(tokenized_train_x), len(tokenized_valid_x), len(tokenized_test_x)


# %%
maxlen = 200
X_train = pad_sequences(tokenized_train_x, maxlen=maxlen)
X_valid = pad_sequences(tokenized_valid_x, maxlen=maxlen)
X_test = pad_sequences(tokenized_test_x, maxlen=maxlen)


# %%
len(X_train), len(X_valid), len(X_test)


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
# ### Clean the data
%load_ext autoreload
%autoreload 2
from utils.preprocessing import clean

clean_train_x = clean(train_x)
clean_valid_x = clean(valid_x)
clean_test_x = clean(test_x)

# %% [markdown]
# ### Stopwords

# %%
%load_ext autoreload
%autoreload 2
from utils.preprocessing import filter_stop_words
filtered_train_x = filter_stop_words(clean_train_x)
filtered_valid_x = filter_stop_words(clean_valid_x)
filtered_test_x = filter_stop_words(clean_test_x)

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(list(filtered_train_x))

# %%
tokenized_train_x = tokenizer.texts_to_sequences(filtered_train_x)
tokenized_valid_x = tokenizer.texts_to_sequences(filtered_valid_x)
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
lemmatized_valid_x = lemmatize(filtered_valid_x)
lemmatized_test_x = lemmatize(filtered_test_x)

tokenizer.fit_on_texts(list(lemmatized_train_x))

# %%
tokenized_train_x = tokenizer.texts_to_sequences(lemmatized_train_x)
tokenized_valid_x = tokenizer.texts_to_sequences(lemmatized_valid_x)
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
stemmed_valid_x = stem(filtered_valid_x)
stemmed_test_x = stem(filtered_test_x)

tokenizer.fit_on_texts(list(stemmed_train_x))

# %%
tokenized_train_x = tokenizer.texts_to_sequences(stemmed_train_x)
tokenized_valid_x = tokenizer.texts_to_sequences(stemmed_valid_x)
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

# %%
print(valid_x[0], "\n")
print(filtered_valid_x[0], "\n")
print(lemmatized_valid_x[0], "\n")
print(stemmed_valid_x[0], "\n")
