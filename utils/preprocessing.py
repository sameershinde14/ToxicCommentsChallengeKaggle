import re
import unicodedata
import inflect

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, sent_tokenize

WORD_CHARACTER_LIMIT = 30


def lemmatize(sentences):
    lemmatizer = WordNetLemmatizer().lemmatize
    lemmatized_sentences = [
        " ".join([
            lemmatizer(word[:WORD_CHARACTER_LIMIT], pos="v")
            for word in sentence.split()
        ])
        for sentence in sentences
    ]

    return lemmatized_sentences


def filter_stop_words(sentences):
    stop_words = set(stopwords.words('english'))

    filtered_sentences = [
        " ".join(
            list(
                filter(
                    lambda word: word not in stop_words,
                    sentence.split()
                )
            )
        )
        for sentence in sentences
    ]

    return filtered_sentences


def stem(sentences):
    stemmer = PorterStemmer().stem
    stemmed_sentences = [
        " ".join([
            stemmer(word[:WORD_CHARACTER_LIMIT])
            for word in sentence.split()
        ])
        for sentence in sentences
    ]

    return stemmed_sentences

def remove_non_ascii(word):
    return unicodedata.normalize('NFKD',word).encode('ascii','ignore').decode('utf-8','ignore')

def remove_punctuations(sentence):
    return re.sub(r'[^\w\s]', '', sentence)

def replace_numbers(sentence):
    return re.sub(r'[0-9]*', '', sentence)

def clean(sentences):
    for index, sentence in enumerate(sentences):
        sentence = replace_numbers(sentence)
        sentence = remove_punctuations(sentence)
        words = word_tokenize(sentence)
        words = map(remove_non_ascii, words)
        words = map(lambda x: x.lower(), words)
        sentences[index] = " ".join(words)

    return sentences