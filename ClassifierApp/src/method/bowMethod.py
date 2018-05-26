import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from src.xml_reader import read_stop_words_list


def invoke(X_train, X_test, y_train, y_test, clf, iterations):
    fit_time_acc = 0
    predict_time_acc = 0
    accuracy_acc = 0
    roc_auc_acc = 0

    for iter_index in range(0, iterations):
        y_pred, fit_time, predict_time, y_score = iter_step(X_train, X_test, y_train, clf)
        fit_time_acc += fit_time
        predict_time_acc += predict_time
        accuracy_acc += accuracy_score(y_test, y_pred)
        # roc_auc_acc += roc_auc_score(target_test, y_score)

    mean_fit_time = fit_time_acc / iterations
    mean_predict_time = predict_time_acc / iterations
    mean_accuracy = accuracy_acc / iterations
    mean_roc_auc_acc = roc_auc_acc / iterations

    return mean_roc_auc_acc, mean_accuracy, mean_fit_time, mean_predict_time


def iter_step(X_train, X_test, y_train, clf):
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

    return y_pred, fit_time, predict_time, y_score
