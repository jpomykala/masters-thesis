import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.xml_reader import read_stop_words_list


def learn_predict(X_train, X_test, y_train, clf):
    stop_words_list = read_stop_words_list("../data/stop_words/list.txt")
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words_list)
    pipeline = Pipeline([
        ('vect', tfidf_vectorizer),
        ('clf', clf)])

    start = time.time()
    pipeline = pipeline.fit(X_train, y_train)
    end = time.time()
    fit_time = (end - start)

    start = time.time()
    y_pred = pipeline.predict(X_test)
    end = time.time()
    predict_time = (end - start)

    # y_score = pipeline.decision_function(data_test)
    y_score = 0

    # y_score = pipeline.score(X_test, y_pred)

    return y_pred, fit_time, predict_time, y_score
