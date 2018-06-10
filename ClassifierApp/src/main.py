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

svm_clf = LinearSVC()
dt_clf = DecisionTreeClassifier()
nb_clf = MultinomialNB()
ft_clf = FastText(dim=7, min_count=15, loss='ns', epoch=200, bucket=200000, word_ngrams=1)

stop_words_list = read_stop_words_list("../data/stop_words/list.txt")
d_vectorizer = TfidfVectorizer(stop_words=stop_words_list)
vectorizer_1 = TfidfVectorizer(stop_words=stop_words_list, ngram_range=(5, 5), analyzer='char')
vectorizer_2 = TfidfVectorizer(stop_words=stop_words_list, ngram_range=(6, 6), analyzer='char')
vectorizer_3 = TfidfVectorizer(stop_words=stop_words_list, ngram_range=(7, 7), analyzer='char')


# ft_clf = GensimFastText(dim=7, min_count=15, loss='ns', epoch=200, bucket=200000, word_ngrams=1)


def show_support_matrix(korpus_name, target_names, ft_support, nb_support, svm_support, dt_support):
    classes = target_names
    data = np.array([ft_support, nb_support, svm_support, dt_support])
    data = np.transpose(data)

    shape = (len(classes), 4)
    matrix_data = data.reshape(shape)

    categories = ['fastText', 'NaiveBayes', 'SVM', 'DecisionTree']

    fig, ax = plt.subplots()
    im = ax.imshow(matrix_data)

    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(classes)))

    ax.set_xticklabels(categories)
    ax.set_yticklabels(classes)

    ax.set_aspect('auto')

    for i in range(len(classes)):
        for j in range(len(categories)):
            text = ax.text(j, i, matrix_data[i, j],
                           ha="center", va="center", color="white")

    title = 'Support - ' + korpus_name
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(plot_save_path + title.lower().replace(' ', '-') + "." + plotFormat, dpi=dpi,
                format=plotFormat)
    plt.show()


def show_confusion_matrix(y_test, y_pred, class_names, title, korpus_name):
    cnf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure()
    title = title + ' - ' + korpus_name
    plot_confusion_matrix(cnf_matrix, classes=class_names, title=title)
    title = title.lower()
    plt.savefig(plot_save_path + 'c-matrix-' + title.lower().replace(' ', '-') + "." + plotFormat, dpi=dpi,
                format=plotFormat, bbox_inches='tight')
    plt.show()


def draw_fit_time_plot(ax_samples, korpus_name):
    plt.plot(ax_samples, ft_fit_times, 'c-+', label="FastText")
    plt.plot(ax_samples, nb_fit_times, 'r-*', label="NaiveBayes")
    plt.plot(ax_samples, svm_fit_times, 'g-^', label="SVM")
    plt.plot(ax_samples, dt_fit_times, 'b-s', label="Decision Tree")
    plt.grid(color='tab:gray', linestyle='-', linewidth=0.15)
    plt.ylabel('time [s]')
    plt.xlabel('number of examples')
    title = 'Fit time - ' + korpus_name
    plt.title(title)
    plt.legend()
    plt.savefig(plot_save_path + title.lower().replace(' ', '-') + "." + plotFormat, dpi=dpi, format=plotFormat)
    plt.show()


def draw_predict_time_plot(ax_samples, korpus_name):
    plt.plot(list(reversed(ax_samples)), list(reversed(ft_predict_times)), 'c-+', label="FastText")
    plt.plot(list(reversed(ax_samples)), list(reversed(nb_predict_times)), 'r-*', label="NaiveBayes")
    plt.plot(list(reversed(ax_samples)), list(reversed(svm_predict_times)), 'g-^', label="SVM")
    plt.plot(list(reversed(ax_samples)), list(reversed(dt_predict_times)), 'b-s', label="Decision Tree")
    plt.grid(color='tab:gray', linestyle='-', linewidth=0.15)
    plt.ylabel('time [s]')
    plt.xlabel('number of examples')
    title = 'Predict time - ' + korpus_name
    plt.title(title)
    plt.legend()
    plt.savefig(plot_save_path + title.lower().replace(' ', '-') + "." + plotFormat, dpi=dpi, format=plotFormat)
    plt.show()


def draw_time_plot(ax_samples, korpus_name):
    plt.plot(ax_samples, ft_times, 'c-+', label="FastText")
    plt.plot(ax_samples, nb_times, 'r-*', label="NaiveBayes")
    plt.plot(ax_samples, svm_times, 'g-^', label="SVM")
    plt.plot(ax_samples, dt_times, 'b-s', label="Decision Tree")
    plt.grid(color='tab:gray', linestyle='-', linewidth=0.15)
    plt.ylabel('time [s]')
    plt.xlabel('number of examples')
    title = 'Total work time - ' + korpus_name
    plt.title(title)
    plt.legend()
    plt.savefig(plot_save_path + title.lower().replace(' ', '-') + "." + plotFormat, dpi=dpi, format=plotFormat)
    plt.show()


def draw_accuracy_plot(ax_samples, korpus_name):
    plt.plot(ax_samples, ft_accuracies, 'c-+', label="FastText")
    plt.plot(ax_samples, nb_accuracies, 'r-*', label="NaiveBayes")
    plt.plot(ax_samples, svm_accuracies, 'g-^', label="SVM")
    plt.plot(ax_samples, dt_accuracies, 'b-s', label="Decision Tree")
    plt.grid(color='tab:gray', linestyle='-', linewidth=0.15)
    plt.ylabel('accuracy')
    plt.xlabel('number of examples')
    title = 'Accuracy - ' + korpus_name
    plt.title(title)
    plt.legend()
    plt.savefig(plot_save_path + title.lower().replace(' ', '-') + "." + plotFormat, dpi=dpi, format=plotFormat)
    plt.show()


def draw_fastText_plot(ax_samples, title):
    plt.plot(ax_samples, fastText_accuracies_1, 'r-*', label="epoch = 5")
    plt.plot(ax_samples, fastText_accuracies_2, 'g-^', label="epoch = 50")
    plt.plot(ax_samples, fastText_accuracies_3, 'b-s', label="epoch = 200")
    plt.plot(ax_samples, fastText_accuracies_4, 'm-h', label="epoch = 500")
    plt.plot(ax_samples, fastText_accuracies_5, 'c-+', label="epoch = 1000")
    plt.grid(color='tab:gray', linestyle='-', linewidth=0.15)
    plt.ylabel('accuracy')
    plt.xlabel('number of examples')
    plt.title(title)
    plt.legend()
    plt.savefig(plot_save_path + title.lower().replace(' ', '-') + "." + plotFormat, dpi=dpi, format=plotFormat)
    plt.show()


def draw_bow_plot(ax_samples, korpus_name):
    plt.plot(ax_samples, bow_1, 'r-*', label="ngram = 5")
    plt.plot(ax_samples, bow_2, 'g-^', label="ngram = 6")
    plt.plot(ax_samples, bow_3, 'b-s', label="ngram = 7")
    plt.grid(color='tab:gray', linestyle='-', linewidth=0.15)
    plt.ylabel('accuracy')
    plt.xlabel('number of examples')
    title = 'BoW - TFIDF - ' + korpus_name + ' - character n-gram'
    plt.title(title)
    plt.legend()
    plt.savefig(plot_save_path + title.lower().replace(' ', '-') + "." + plotFormat, dpi=dpi, format=plotFormat)
    plt.show()


def accuracy_time_report(train_sizes, iterations, korpus_path, korpus_name):
    global bow_1, bow_2, bow_3, train_samples_array, test_samples_array, ft_accuracies, nb_accuracies, svm_accuracies, dt_accuracies, ft_roc_auc_score_array, nb_roc_auc_score_array, svm_roc_auc_score_array, dt_roc_auc_score_array, ft_fit_times, nb_fit_times, svm_fit_times, dt_fit_times, ft_predict_times, nb_predict_times, svm_predict_times, dt_predict_times, ft_times, nb_times, svm_times, dt_times
    train_samples_array = []
    test_samples_array = []
    ft_accuracies = []
    nb_accuracies = []
    svm_accuracies = []
    dt_accuracies = []
    ft_roc_auc_score_array = []
    nb_roc_auc_score_array = []
    svm_roc_auc_score_array = []
    dt_roc_auc_score_array = []
    ft_fit_times = []
    nb_fit_times = []
    svm_fit_times = []
    dt_fit_times = []
    ft_predict_times = []
    nb_predict_times = []
    svm_predict_times = []
    dt_predict_times = []
    ft_times = []
    nb_times = []
    svm_times = []
    dt_times = []

    bow_1 = []
    bow_2 = []
    bow_3 = []

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
        test_samples_array.append(test_samples_count)

        print('Calculating... train:', str(train_samples_count), '| test:', str(test_samples_count))

        # learning curve
        ft_accuracy, ft_fit_time, ft_predict_time, ft_roc_auc = calc_wrapper.start_test(
            iterations, y_test, fastTextMethod.learn_predict, (X_train, X_test, y_train, ft_clf))

        nb_accuracy, nb_fit_time, nb_predict_time, nb_roc_auc = calc_wrapper.start_test(
            iterations, y_test, bowMethod.learn_predict, (X_train, X_test, y_train, nb_clf, d_vectorizer))

        svm_accuracy, svm_fit_time, svm_predict_time, svm_roc_auc = calc_wrapper.start_test(
            iterations, y_test, bowMethod.learn_predict, (X_train, X_test, y_train, svm_clf, d_vectorizer))

        dt_accuracy, dt_fit_time, dt_predict_time, dt_roc_auc = calc_wrapper.start_test(
            iterations, y_test, bowMethod.learn_predict, (X_train, X_test, y_train, dt_clf, d_vectorizer))

        nb_accuracy_2, nb_fit_time, nb_predict_time, nb_roc_auc = calc_wrapper.start_test(
            iterations, y_test, bowMethod.learn_predict, (X_train, X_test, y_train, nb_clf, vectorizer_2))

        svm_accuracy_2, svm_fit_time, svm_predict_time, svm_roc_auc = calc_wrapper.start_test(
            iterations, y_test, bowMethod.learn_predict, (X_train, X_test, y_train, svm_clf, vectorizer_2))

        dt_accuracy_2, dt_fit_time, dt_predict_time, dt_roc_auc = calc_wrapper.start_test(
            iterations, y_test, bowMethod.learn_predict, (X_train, X_test, y_train, dt_clf, vectorizer_2))

        nb_accuracy_3, nb_fit_time, nb_predict_time, nb_roc_auc = calc_wrapper.start_test(
            iterations, y_test, bowMethod.learn_predict, (X_train, X_test, y_train, nb_clf, vectorizer_3))

        svm_accuracy_3, svm_fit_time, svm_predict_time, svm_roc_auc = calc_wrapper.start_test(
            iterations, y_test, bowMethod.learn_predict, (X_train, X_test, y_train, svm_clf, vectorizer_3))

        dt_accuracy_3, dt_fit_time, dt_predict_time, dt_roc_auc = calc_wrapper.start_test(
            iterations, y_test, bowMethod.learn_predict, (X_train, X_test, y_train, dt_clf, vectorizer_3))

        bow_1.append((nb_accuracy+svm_accuracy+dt_accuracy)/3)
        bow_2.append((nb_accuracy_2+svm_accuracy_2+dt_accuracy_2)/3)
        bow_3.append((nb_accuracy_3+svm_accuracy_3+dt_accuracy_3)/3)

        ft_fit_times.append(ft_fit_time)
        nb_fit_times.append(nb_fit_time)
        svm_fit_times.append(svm_fit_time)
        dt_fit_times.append(dt_fit_time)

        ft_predict_times.append(ft_predict_time)
        nb_predict_times.append(nb_predict_time)
        svm_predict_times.append(svm_predict_time)
        dt_predict_times.append(dt_predict_time)

        ft_times.append(ft_fit_time + ft_predict_time)
        nb_times.append(nb_fit_time + nb_predict_time)
        svm_times.append(svm_fit_time + svm_predict_time)
        dt_times.append(dt_fit_time + dt_predict_time)

        ft_accuracies.append(ft_accuracy)
        nb_accuracies.append(nb_accuracy)
        svm_accuracies.append(svm_accuracy)
        dt_accuracies.append(dt_accuracy)

        ft_roc_auc_score_array.append(ft_roc_auc)
        nb_roc_auc_score_array.append(nb_roc_auc)
        svm_roc_auc_score_array.append(svm_roc_auc)
        dt_roc_auc_score_array.append(dt_roc_auc)

        # draw plots
        # draw_accuracy_plot(train_samples_array, korpus_name)
        # draw_fit_time_plot(train_samples_array, korpus_name)
        # draw_predict_time_plot(test_samples_array, korpus_name)
        # draw_time_plot(train_samples_array, korpus_name)

        draw_bow_plot(train_samples_array, korpus_name)

        step += 1
        print("Finished:", format((step / len(train_sizes)) * 100, '.2f') + "%")


def cls_report(korpus_path, korpus_name):
    train_size = 0.6
    X_test, X_train, y_test, y_train, files_data = load_string_korpus(korpus_path, train_size)
    target_names = files_data.target_names
    iter = 1

    calc_wrapper.report_data(
        'fastText - ' + korpus_name, iter, y_test, target_names, fastTextMethod.learn_predict,
        (X_train, X_test, y_train, ft_clf))
    calc_wrapper.report_data(
        'NaiveBayes - ' + korpus_name, iter, y_test, target_names, bowMethod.learn_predict,
        (X_train, X_test, y_train, nb_clf, d_vectorizer))
    calc_wrapper.report_data(
        'SVM - ' + korpus_name, iter, y_test, target_names, bowMethod.learn_predict,
        (X_train, X_test, y_train, svm_clf, d_vectorizer))
    calc_wrapper.report_data(
        'DecisionTree - ' + korpus_name, iter, y_test, target_names, bowMethod.learn_predict,
        (X_train, X_test, y_train, dt_clf, d_vectorizer))

    # confusion matrix
    y_pred_ft, fit_time, predict_time, y_score = fastTextMethod.learn_predict(X_train, X_test, y_train, ft_clf)
    show_confusion_matrix(y_test, y_pred_ft, target_names, 'fastText', korpus_name)

    y_pred_nb, fit_time, predict_time, y_score = bowMethod.learn_predict(X_train, X_test, y_train, nb_clf, d_vectorizer)
    show_confusion_matrix(y_test, y_pred_nb, target_names, 'NaiveBayes', korpus_name)

    y_pred_svm, fit_time, predict_time, y_score = bowMethod.learn_predict(X_train, X_test, y_train, svm_clf,
                                                                          d_vectorizer)
    show_confusion_matrix(y_test, y_pred_svm, target_names, 'SVM', korpus_name)

    y_pred_dt, fit_time, predict_time, y_score = bowMethod.learn_predict(X_train, X_test, y_train, dt_clf, d_vectorizer)
    show_confusion_matrix(y_test, y_pred_dt, target_names, 'DecisionTree', korpus_name)


def load_string_korpus(korpus_path, train_size):
    files_data = load_files(korpus_path, encoding='utf-8')
    X_train, X_test, y_train, y_test = train_test_split(
        files_data.data,
        files_data.target,
        train_size=train_size,
        test_size=1 - train_size)
    return X_test, X_train, y_test, y_train, files_data


def start_tests():
    iterations_wiki = 2
    iterations_articles = 2
    train_sizes_wiki = np.arange(0.01, 0.51, 0.06)
    train_sizes_articles = np.arange(0.01, 0.51, 0.03)

    data_sets = [
        # ('Wikipedia', "../data/wiki/lemma", iterations_wiki, train_sizes_wiki),
        ('Articles', "../data/korpus/lemma", iterations_articles, train_sizes_articles),
        # ('Wikipedia (nouns)', "../data/wiki/noun", iterations_wiki, train_sizes_wiki),
        # ('Articles (nouns)', "../data/korpus/noun", iterations_articles, train_sizes_articles),
    ]

    for korpus_name, korpus_path, iter_size, train_size in data_sets:
        print('Korpus name: %s' % korpus_name)
        accuracy_time_report(train_size, iter_size, korpus_path, korpus_name)
        cls_report(korpus_path, korpus_name)


start_tests()
