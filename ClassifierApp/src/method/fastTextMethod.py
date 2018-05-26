from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, auc, average_precision_score
from sklearn.pipeline import Pipeline
import time
import numpy as np


text_category = "../data/korpus/text"
ccl_category = "../data/korpus/ccl"
lemma_category = "../data/korpus/lemma"
raw_files = "../data/korpus/raw"


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
    pipeline = Pipeline([
        ('clf', clf)
    ])

    data_train_tuple = []
    for dt in X_train:
        data_train_tuple.append(tuple(dt.split()))

    start = time.time()
    pipeline = pipeline.fit(data_train_tuple, y_train)
    end = time.time()
    fit_time = (end - start)

    data_test_tuple = []
    for dt in X_test:
        data_test_tuple.append(tuple(dt.split()))

    start = time.time()
    predicted = pipeline.predict(data_test_tuple)
    end = time.time()
    predict_time = (end - start)

    tmp_arr = []
    for p in predicted:
        if p is None:
            tmp_arr.append(0)
        else:
            tmp_arr.append(int(p))

    y_pred = np.asarray(tmp_arr, dtype=np.int64)

    # y_score = pipeline.decision_function(data_test_tuple)
    y_score = 0

    return y_pred, fit_time, predict_time, y_score


