import matplotlib.pyplot as plt
import numpy as np
from shallowlearn.models import FastText
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src import calc_wrapper
from src.confusion import plot_confusion_matrix
from src.consts import plotFormat, dpi, plot_save_path
from src.method import bowMethod
from src.method import fastTextMethod
from src.xml_reader import read_stop_words_list

n_classes = 34

ft_ngram_clf_1 = FastText(dim=n_classes, min_count=15, loss='ns', epoch=100, bucket=200000, word_ngrams=1)
ft_ngram_clf_2 = FastText(dim=n_classes, min_count=15, loss='ns', epoch=100, bucket=200000, word_ngrams=2)
ft_ngram_clf_3 = FastText(dim=n_classes, min_count=15, loss='ns', epoch=100, bucket=200000, word_ngrams=3)

ft_epoch_clf_1 = FastText(dim=n_classes, min_count=15, loss='ns', epoch=5, bucket=200000, word_ngrams=1)
ft_epoch_clf_2 = FastText(dim=n_classes, min_count=15, loss='ns', epoch=20, bucket=200000, word_ngrams=1)
ft_epoch_clf_3 = FastText(dim=n_classes, min_count=15, loss='ns', epoch=50, bucket=200000, word_ngrams=1)
ft_epoch_clf_4 = FastText(dim=n_classes, min_count=15, loss='ns', epoch=100, bucket=200000, word_ngrams=1)
ft_epoch_clf_5 = FastText(dim=n_classes, min_count=15, loss='ns', epoch=200, bucket=200000, word_ngrams=1)

ft_min_count_clf_1 = FastText(dim=n_classes, min_count=0, loss='ns', epoch=200, bucket=200000, word_ngrams=1)
ft_min_count_clf_2 = FastText(dim=n_classes, min_count=5, loss='ns', epoch=200, bucket=200000, word_ngrams=1)
ft_min_count_clf_3 = FastText(dim=n_classes, min_count=10, loss='ns', epoch=200, bucket=200000, word_ngrams=1)
ft_min_count_clf_4 = FastText(dim=n_classes, min_count=20, loss='ns', epoch=200, bucket=200000, word_ngrams=1)
ft_min_count_clf_5 = FastText(dim=n_classes, min_count=100, loss='ns', epoch=200, bucket=200000, word_ngrams=1)

ft_loss_clf_1 = FastText(dim=n_classes, min_count=100, loss='ns', epoch=200, bucket=200000, word_ngrams=1)
ft_loss_clf_2 = FastText(dim=n_classes, min_count=100, loss='hs', epoch=200, bucket=200000, word_ngrams=1)
ft_loss_clf_3 = FastText(dim=n_classes, min_count=100, loss='softmax', epoch=200, bucket=200000, word_ngrams=1)


def draw_epoch_plot(ax_samples, korpus_name):
    plt.plot(ax_samples, ft_epoch_result_1, 'r-*', label="epoch = 5")
    plt.plot(ax_samples, ft_epoch_result_2, 'g-^', label="epoch = 20")
    plt.plot(ax_samples, ft_epoch_result_3, 'b-s', label="epoch = 50")
    plt.plot(ax_samples, ft_epoch_result_4, 'm-h', label="epoch = 100")
    plt.plot(ax_samples, ft_epoch_result_5, 'c-+', label="epoch = 200")
    plt.grid(color='tab:gray', linestyle='-', linewidth=0.15)
    plt.ylabel('dokładność')
    plt.xlabel('liczba próbek')
    title = 'fastText - epoch - ' + korpus_name
    plt.title(title)
    plt.legend()
    plt.savefig(plot_save_path + title.lower().replace(' ', '-') + "." + plotFormat, dpi=dpi, format=plotFormat)
    plt.show()


def draw_ngram_plot(ax_samples, korpus_name):
    plt.plot(ax_samples, ft_ngram_result_1, 'r-*', label="ngram = 1")
    plt.plot(ax_samples, ft_ngram_result_2, 'g-^', label="ngram = 2")
    plt.plot(ax_samples, ft_ngram_result_3, 'b-s', label="ngram = 3")
    plt.grid(color='tab:gray', linestyle='-', linewidth=0.15)
    plt.ylabel('dokładność')
    plt.xlabel('liczba próbek')
    title = 'fastText - ngram - ' + korpus_name
    plt.title(title)
    plt.legend()
    plt.savefig(plot_save_path + title.lower().replace(' ', '-') + "." + plotFormat, dpi=dpi, format=plotFormat)
    plt.show()


def draw_min_count_plot(ax_samples, korpus_name):
    plt.plot(ax_samples, ft_min_count_result_1, 'r-*', label="min_count = 0")
    plt.plot(ax_samples, ft_min_count_result_2, 'g-^', label="min_count = 5")
    plt.plot(ax_samples, ft_min_count_result_3, 'b-s', label="min_count = 10")
    plt.plot(ax_samples, ft_min_count_result_4, 'm-h', label="min_count = 20")
    plt.plot(ax_samples, ft_min_count_result_5, 'c-+', label="min_count = 100")
    plt.grid(color='tab:gray', linestyle='-', linewidth=0.15)
    plt.ylabel('dokładność')
    plt.xlabel('liczba próbek')
    title = 'fastText - min_count - ' + korpus_name
    plt.title(title)
    plt.legend()
    plt.savefig(plot_save_path + title.lower().replace(' ', '-') + "." + plotFormat, dpi=dpi, format=plotFormat)
    plt.show()


def draw_loss_plot(ax_samples, korpus_name):
    plt.plot(ax_samples, ft_loss_result_1, 'r-*', label="loss = ns")
    plt.plot(ax_samples, ft_loss_result_2, 'g-^', label="loss = hs")
    plt.plot(ax_samples, ft_loss_result_3, 'b-s', label="loss = softmax")
    plt.grid(color='tab:gray', linestyle='-', linewidth=0.15)
    plt.ylabel('dokładność')
    plt.xlabel('liczba próbek')
    title = 'fastText - loss - ' + korpus_name
    plt.title(title)
    plt.legend()
    plt.savefig(plot_save_path + title.lower().replace(' ', '-') + "." + plotFormat, dpi=dpi, format=plotFormat)
    plt.show()


def accuracy_time_report(train_sizes, iterations, korpus_path, korpus_name):
    global train_samples_array, test_samples_array, ft_loss_result_1, ft_loss_result_2, ft_loss_result_3, ft_min_count_result_1, ft_min_count_result_2, ft_min_count_result_3, ft_min_count_result_4, ft_min_count_result_5, ft_ngram_result_1, ft_ngram_result_2, ft_ngram_result_3, ft_epoch_result_1, ft_epoch_result_2, ft_epoch_result_3, ft_epoch_result_4, ft_epoch_result_5;

    train_samples_array = []
    test_samples_array = []

    ft_loss_result_1 = []
    ft_loss_result_2 = []
    ft_loss_result_3 = []

    ft_min_count_result_1 = []
    ft_min_count_result_2 = []
    ft_min_count_result_3 = []
    ft_min_count_result_4 = []
    ft_min_count_result_5 = []

    ft_ngram_result_1 = []
    ft_ngram_result_2 = []
    ft_ngram_result_3 = []

    ft_epoch_result_1 = []
    ft_epoch_result_2 = []
    ft_epoch_result_3 = []
    ft_epoch_result_4 = []
    ft_epoch_result_5 = []

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

        #ngram
        ft_ngram_1 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, ft_ngram_clf_1)
        ft_ngram_2 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, ft_ngram_clf_2)
        ft_ngram_3 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, ft_ngram_clf_3)

        ft_ngram_result_1.append(ft_ngram_1)
        ft_ngram_result_2.append(ft_ngram_2)
        ft_ngram_result_3.append(ft_ngram_3)

        #epoch
        ft_epoch_1 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, ft_epoch_clf_1)
        ft_epoch_2 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, ft_epoch_clf_2)
        ft_epoch_3 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, ft_epoch_clf_3)
        ft_epoch_4 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, ft_epoch_clf_4)
        ft_epoch_5 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, ft_epoch_clf_5)

        ft_epoch_result_1.append(ft_epoch_1)
        ft_epoch_result_2.append(ft_epoch_2)
        ft_epoch_result_3.append(ft_epoch_3)
        ft_epoch_result_4.append(ft_epoch_4)
        ft_epoch_result_5.append(ft_epoch_5)
        
        #mincount
        ft_min_count_1 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, ft_min_count_clf_1)
        ft_min_count_2 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, ft_min_count_clf_2)
        ft_min_count_3 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, ft_min_count_clf_3)
        ft_min_count_4 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, ft_min_count_clf_4)
        ft_min_count_5 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, ft_min_count_clf_5)

        ft_min_count_result_1.append(ft_min_count_1)
        ft_min_count_result_2.append(ft_min_count_2)
        ft_min_count_result_3.append(ft_min_count_3)
        ft_min_count_result_4.append(ft_min_count_4)
        ft_min_count_result_5.append(ft_min_count_5)
        
        #hs
        ft_loss_1 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, ft_loss_clf_1)
        ft_loss_2 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, ft_loss_clf_2)
        ft_loss_3 = simple_wrapper(X_test, X_train, iterations, y_test, y_train, ft_loss_clf_3)

        ft_loss_result_1.append(ft_loss_1)
        ft_loss_result_2.append(ft_loss_2)
        ft_loss_result_3.append(ft_loss_3)

        # draw plots
        draw_epoch_plot(train_samples_array, korpus_name)
        draw_ngram_plot(train_samples_array, korpus_name)
        draw_min_count_plot(train_samples_array, korpus_name)
        draw_loss_plot(train_samples_array, korpus_name)

        step += 1
        print("Finished:", format((step / len(train_sizes)) * 100, '.2f') + "%")


def simple_wrapper(X_test, X_train, iterations, y_test, y_train, clf):
    ft_accuracy, ft_fit_time, ft_predict_time, ft_roc_auc = calc_wrapper.start_test(
        iterations, y_test, fastTextMethod.learn_predict, (X_train, X_test, y_train, clf))
    return ft_accuracy


def start_tests():
    iterations_wiki = 10
    iterations_articles = 10
    train_sizes_wiki = np.arange(0.01, 0.51, 0.06)
    train_sizes_articles = np.arange(0.01, 0.51, 0.03)

    data_sets = [
        ('Wikipedia', "../data/wiki/lemma", iterations_wiki, train_sizes_wiki),
        ('Artykuły', "../data/korpus/lemma", iterations_articles, train_sizes_articles),
        ('Wikipedia (rzeczowniki)', "../data/wiki/noun", iterations_wiki, train_sizes_wiki),
        ('Artykuły (rzeczowniki)', "../data/korpus/noun", iterations_articles, train_sizes_articles),
    ]

    for korpus_name, korpus_path, iter_size, train_size in data_sets:
        print('Korpus name: %s' % korpus_name)
        accuracy_time_report(train_size, iter_size, korpus_path, korpus_name)


start_tests()
