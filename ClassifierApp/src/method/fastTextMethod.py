from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
import time
import numpy as np

text_category = "../data/korpus/text"
ccl_category = "../data/korpus/ccl"
lemma_category = "../data/korpus/lemma"
raw_files = "../data/korpus/raw"


def invoke(data_train, data_test, target_train, target_test, clf, iterations):
    fit_time_acc = 0
    predict_time_acc = 0
    accuracy_acc = 0
    f1_acc = 0

    for iter_index in range(0, iterations):
        predicted, fit_time, predict_time = iter_step(data_train, data_test, target_train, clf)
        fit_time_acc += fit_time
        predict_time_acc += predict_time

        tmp_arr = []
        for p in predicted:
            if p is None:
                tmp_arr.append(0)
            else:
                tmp_arr.append(int(p))

        predicted_arr = np.asarray(tmp_arr, dtype=np.int64)
        accuracy_acc += accuracy_score(target_test, predicted_arr)
        f1_acc += f1_score(target_test, predicted_arr, average='micro')

    mean_fit_time = fit_time_acc / iterations
    mean_predict_time = predict_time_acc / iterations
    mean_accuracy = accuracy_acc / iterations
    mean_f1_acc = f1_acc / iterations

    return mean_f1_acc, mean_accuracy, mean_fit_time, mean_predict_time


def iter_step(data_train, data_test, target_train, clf):
    pipeline = Pipeline([
        ('clf', clf)
    ])

    data_train_tuple = []
    for dt in data_train:
        data_train_tuple.append(tuple(dt.split()))

    start = time.time()
    pipeline = pipeline.fit(data_train_tuple, target_train)
    end = time.time()
    fit_time = (end - start)

    data_test_tuple = []
    for dt in data_test:
        data_test_tuple.append(tuple(dt.split()))

    start = time.time()
    predicted = pipeline.predict(data_test_tuple)
    end = time.time()
    predict_time = (end - start)

    return predicted, fit_time, predict_time


