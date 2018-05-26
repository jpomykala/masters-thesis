from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
import time
from src.xml_reader import read_stop_words_list


def invoke(data_train, data_test, target_train, target_test, clf, iterations):
    fit_time_acc = 0
    predict_time_acc = 0
    accuracy_acc = 0
    f1_acc = 0

    for iter_index in range(0, iterations):
        predicted, fit_time, predict_time = iter_step(data_train, data_test, target_train, clf)
        fit_time_acc += fit_time
        predict_time_acc += predict_time
        accuracy_acc += accuracy_score(target_test, predicted)
        f1_acc += f1_score(target_test, predicted, average='micro')

    mean_fit_time = fit_time_acc / iterations
    mean_predict_time = predict_time_acc / iterations
    mean_accuracy = accuracy_acc / iterations
    mean_f1_acc = f1_acc / iterations

    return mean_f1_acc, mean_accuracy, mean_fit_time, mean_predict_time


def iter_step(data_train, data_test, target_train, classifier):
    stop_words_list = read_stop_words_list("../data/stop_words/list.txt")
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words_list)
    pipeline = Pipeline([
        ('vect', tfidf_vectorizer),
        ('clf', classifier)])

    start = time.time()
    pipeline = pipeline.fit(data_train, target_train)
    end = time.time()
    fit_time = (end - start)

    start = time.time()
    predicted = pipeline.predict(data_test)
    end = time.time()
    predict_time = (end - start)

    return predicted, fit_time, predict_time
