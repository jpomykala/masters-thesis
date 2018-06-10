import time

from sklearn.pipeline import Pipeline


def learn_predict(X_train, X_test, y_train, clf, vect):
    pipeline = Pipeline([
        ('vect', vect),
        ('clf', clf)])

    start = time.time()
    pipeline = pipeline.fit(X_train, y_train)
    end = time.time()
    fit_time = (end - start)

    start = time.time()
    y_pred = pipeline.predict(X_test)
    end = time.time()
    predict_time = (end - start)
    return y_pred, fit_time, predict_time
