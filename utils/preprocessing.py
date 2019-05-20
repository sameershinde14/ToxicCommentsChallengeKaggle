from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
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
