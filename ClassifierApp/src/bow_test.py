import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

module_path = os.path.abspath(os.getcwd()).replace('/src', '')

if module_path not in sys.path:
    sys.path.append(module_path)
from src import calc_wrapper
from src.consts import plotFormat, dpi, plot_save_path
from src.method import bowMethod
from src.xml_reader import read_stop_words_list

svm_clf = LinearSVC()
dt_clf = DecisionTreeClassifier()
nb_clf = MultinomialNB()

stop_words_list = read_stop_words_list()
vectorizer_word_1 = TfidfVectorizer(stop_words=stop_words_list)
vectorizer_word_2 = TfidfVectorizer(stop_words=stop_words_list, ngram_range=(2, 2))
vectorizer_word_3 = TfidfVectorizer(stop_words=stop_words_list, ngram_range=(3, 3))

vectorizer_char_1 = TfidfVectorizer(stop_words=stop_words_list, ngram_range=(5, 5), analyzer='char')
vectorizer_char_2 = TfidfVectorizer(stop_words=stop_words_list, ngram_range=(6, 6), analyzer='char')
vectorizer_char_3 = TfidfVectorizer(stop_words=stop_words_list, ngram_range=(7, 7), analyzer='char')


def draw_bow_word_plot(ax_samples, korpus_name):
    plt.plot(ax_samples, bow_word_1, 'r-*', label="ngram = 1")
    plt.plot(ax_samples, bow_word_2, 'g-^', label="ngram = 2")
    plt.plot(ax_samples, bow_word_3, 'b-s', label="ngram = 3")
    plt.grid(color='tab:gray', linestyle='-', linewidth=0.15)
    plt.ylabel('dokładność')
    plt.xlabel('liczba próbek')
    title = 'Bag-Of-Words - tfidf - ' + korpus_name + ' - słowa - n-gram'
    plt.title(title)
    plt.legend()
    plt.savefig(plot_save_path + title.lower().replace(' ', '-').replace('ł', 'l') + "." + plotFormat, dpi=dpi,
                format=plotFormat)
    plt.show()


def draw_bow_char_plot(ax_samples, korpus_name):
    plt.plot(ax_samples, bow_char_1, 'r-*', label="ngram = 5")
    plt.plot(ax_samples, bow_char_2, 'g-^', label="ngram = 6")
    plt.plot(ax_samples, bow_char_3, 'b-s', label="ngram = 7")
    plt.grid(color='tab:gray', linestyle='-', linewidth=0.15)
    plt.ylabel('dokładność')
    plt.xlabel('liczba próbek')
    title = 'Bag-Of-Words - tfidf - ' + korpus_name + ' - znaki - n-gram'
    plt.title(title)
    plt.legend()
    plt.savefig(plot_save_path + title.lower().replace(' ', '-').replace('ł', 'l') + "." + plotFormat, dpi=dpi,
                format=plotFormat)
    plt.show()


def test_bow(train_sizes, iterations, korpus_path, korpus_name):
    global bow_char_1, bow_char_2, bow_char_3, bow_word_1, bow_word_2, bow_word_3

    train_samples_array = []
    bow_char_1 = []
    bow_char_2 = []
    bow_char_3 = []

    bow_word_1 = []
    bow_word_2 = []
    bow_word_3 = []

    files_data = load_files(korpus_path, encoding='utf-8')

    step = 0  # do obliczania % ukonczenia
    for train_size in train_sizes:
        X_train, X_test, y_train, y_test = train_test_split(
            files_data.data,
            files_data.target,
            train_size=train_size,
            test_size=1 - train_size)

        test_samples_count = len(X_test)
        train_samples_count = len(X_train)
        train_samples_array.append(train_samples_count)

        print('Calculating... train:', str(train_samples_count), '| test:', str(test_samples_count))

        # word
        nb_accuracy_word_1 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, nb_clf, vectorizer_word_1)
        nb_accuracy_word_2 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, nb_clf, vectorizer_word_2)
        nb_accuracy_word_3 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, nb_clf, vectorizer_word_3)

        svm_accuracy_word_1 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, svm_clf, vectorizer_word_1)
        svm_accuracy_word_2 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, svm_clf, vectorizer_word_2)
        svm_accuracy_word_3 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, svm_clf, vectorizer_word_3)

        dt_accuracy_word_1 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, dt_clf, vectorizer_word_1)
        dt_accuracy_word_2 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, dt_clf, vectorizer_word_2)
        dt_accuracy_word_3 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, dt_clf, vectorizer_word_3)

        bow_word_1.append((nb_accuracy_word_1 + svm_accuracy_word_1 + dt_accuracy_word_1) / 3)
        bow_word_2.append((nb_accuracy_word_2 + svm_accuracy_word_2 + dt_accuracy_word_2) / 3)
        bow_word_3.append((nb_accuracy_word_3 + svm_accuracy_word_3 + dt_accuracy_word_3) / 3)

        # word
        nb_accuracy_char_1 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, nb_clf, vectorizer_char_1)
        nb_accuracy_char_2 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, nb_clf, vectorizer_char_2)
        nb_accuracy_char_3 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, nb_clf, vectorizer_char_3)

        svm_accuracy_char_1 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, svm_clf, vectorizer_char_1)
        svm_accuracy_char_2 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, svm_clf, vectorizer_char_2)
        svm_accuracy_char_3 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, svm_clf, vectorizer_char_3)

        dt_accuracy_char_1 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, dt_clf, vectorizer_char_1)
        dt_accuracy_char_2 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, dt_clf, vectorizer_char_2)
        dt_accuracy_char_3 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, dt_clf, vectorizer_char_3)

        bow_char_1.append((nb_accuracy_char_1 + svm_accuracy_char_1 + dt_accuracy_char_1) / 3)
        bow_char_2.append((nb_accuracy_char_2 + svm_accuracy_char_2 + dt_accuracy_char_2) / 3)
        bow_char_3.append((nb_accuracy_char_3 + svm_accuracy_char_3 + dt_accuracy_char_3) / 3)

        # draw plots
        draw_bow_char_plot(train_samples_array, korpus_name)
        draw_bow_word_plot(train_samples_array, korpus_name)

        step += 1
        print("Finished:", format((step / len(train_sizes)) * 100, '.2f') + "%")


def simple_wrapper(X_test, X_train, iterations, y_test, y_train, clf, vect):
    accuracy, dt_fit_time, dt_predict_time = calc_wrapper.start_test(
        iterations, y_test, bowMethod.learn_predict, (X_train, X_test, y_train, clf, vect))
    return accuracy


def start_tests():
    iterations_wiki = 2
    iterations_articles = 2
    train_sizes_wiki = np.arange(0.01, 0.51, 0.06)
    train_sizes_articles = np.arange(0.01, 0.51, 0.03)

    wiki_data_sets = [('Wikipedia (rzeczowniki)', "../data/wiki/noun", iterations_wiki, train_sizes_wiki),
                      # ('Wikipedia', "../data/wiki/lemma", iterations_wiki, train_sizes_wiki),
                      ]

    article_data_sets = [('Artykuły', "../data/korpus/lemma", iterations_articles, train_sizes_articles),
                         ('Artykuły (rzeczowniki)', "../data/korpus/noun", iterations_articles, train_sizes_articles)]

    data_sets = []

    argument_data_set = sys.argv[1:]
    if 'w' in argument_data_set:
        print("loading Wikipedia data set only")
        data_sets = wiki_data_sets

    if 'a' in argument_data_set:
        print("loading Articles data set only")
        data_sets = article_data_sets

    if len(sys.argv) < 2:
        print("loading full data set")
        data_sets = wiki_data_sets + article_data_sets

    for korpus_name, korpus_path, iter_size, train_size in data_sets:
        print('Korpus name: %s' % korpus_name)
        test_bow(train_size, iter_size, korpus_path, korpus_name)


start_tests()
