import pickle


def load_pickles(stem=False):
    if stem:
        train_x = pickle.load(open("pickle/stemmed_train_x.pickle", "rb"))
        valid_x = pickle.load(open("pickle/stemmed_valid_x.pickle", "rb"))
        test_x = pickle.load(open("pickle/stemmed_test_x.pickle", "rb"))
    else:
        train_x = pickle.load(open("pickle/lemmatized_train_x.pickle", "rb"))
        valid_x = pickle.load(open("pickle/lemmatized_valid_x.pickle", "rb"))
        test_x = pickle.load(open("pickle/lemmatized_test_x.pickle", "rb"))

    train_y = pickle.load(open("pickle/train_y.pickle", "rb"))
    valid_y = pickle.load(open("pickle/valid_y.pickle", "rb"))
    test_y = pickle.load(open("pickle/test_y.pickle", "rb"))

    return (
        train_x,
        valid_x,
        test_x,
        train_y,
        valid_y,
        test_y
    )

