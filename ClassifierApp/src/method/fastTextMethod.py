import time

import numpy as np
from sklearn.pipeline import Pipeline

text_category = "../data/korpus/text"
ccl_category = "../data/korpus/ccl"
lemma_category = "../data/korpus/lemma"
raw_files = "../data/korpus/raw"


def learn_predict(X_train, X_test, y_train, clf):
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


