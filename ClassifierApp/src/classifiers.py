from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_classifiers():
    knn = ("kNN", KNeighborsClassifier(8))
    svm = ("SVM", SVC(kernel="linear", C=0.025))
    svc_gamma = ("nonLinearSVC", SVC(gamma=2, C=1))
    tree_classifier = ("DecisionTree", DecisionTreeClassifier(max_depth=10))
    forest_classifier = ("randomForest", RandomForestClassifier(max_depth=10, n_estimators=10, max_features=8))
    mlp_classifier = ("mlpc", MLPClassifier(alpha=1))
    ada_boost_classifier = ("adaBoost", AdaBoostClassifier())
    multinomial_nb = ("MultinomialNB", MultinomialNB())
    bernoulli_nb = ("BernoulliNB", BernoulliNB())
    sgd_classifier = SGDClassifier(loss='hinge', penalty='l2',
                                   alpha=1e-3, random_state=42,
                                   max_iter=5, tol=None)
    perceptron = ("Perceptron", Perceptron(max_iter=5, tol=None))
    classifiers = [
        multinomial_nb,
        svm,
        tree_classifier,
    ]
    return classifiers
