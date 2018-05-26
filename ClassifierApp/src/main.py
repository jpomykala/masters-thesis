import matplotlib.pyplot as plt
from shallowlearn.models import FastText
from sklearn.datasets import load_files
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.confusion import plot_confusion_matrix
from src.method import bowMethod
from src.method import fastTextMethod

#lemma_dir = "../data/wiki/lemma"
lemma_dir = "../data/korpus/lemma"

#korpus_name = "Wikipedia"
korpus_name = "Articles"

plotFormat = "eps"
dpi = None
# train_sizes = np.arange(0.01, 0.51, 0.03)
train_sizes = [0.7]
# train_sizes = [0.1]
iterations = 1 # liczba przebiegów
plot_save_path = "/Users/evelan/Desktop/mgr.nosync/"


def show_confusion_matrix(y_test, y_pred, class_names, title):
    cnf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure()
    title = title + ' - ' + korpus_name
    plot_confusion_matrix(cnf_matrix, classes=class_names, title=title)
    title = title.lower()
    plt.savefig(plot_save_path + title + "." + plotFormat, dpi=dpi, format=plotFormat, bbox_inches='tight')
    plt.show()


def draw_fit_time_plot():
    plt.plot(train_samples_array, ft_fit_times, 'c-+', label="FastText")
    plt.plot(train_samples_array, nb_fit_times, 'r-*', label="NaiveBayes")
    plt.plot(train_samples_array, svm_fit_times, 'g-^', label="SVM")
    plt.plot(train_samples_array, dt_fit_times, 'b-s', label="Decision Tree")
    plt.grid(color='tab:gray', linestyle='-', linewidth=0.15)
    plt.ylabel('time [s]')
    plt.xlabel('number of examples')
    title = 'Fit time - ' + korpus_name
    plt.title(title)
    plt.legend()
    plt.savefig(plot_save_path+title + "." + plotFormat, dpi=dpi, format=plotFormat)
    plt.show()


def draw_predict_time_plot():
    plt.plot(train_samples_array, ft_predict_times, 'c-+', label="FastText")
    plt.plot(train_samples_array, nb_predict_times, 'r-*', label="NaiveBayes")
    plt.plot(train_samples_array, svm_predict_times, 'g-^', label="SVM")
    plt.plot(train_samples_array, dt_predict_times, 'b-s', label="Decision Tree")
    plt.grid(color='tab:gray', linestyle='-', linewidth=0.15)
    plt.ylabel('time [s]')
    plt.xlabel('number of examples')
    title = 'Predict time - ' + korpus_name
    plt.title(title)
    plt.legend()
    plt.savefig(plot_save_path+title + "." + plotFormat, dpi=dpi, format=plotFormat)
    plt.show()


def draw_time_plot():
    plt.plot(train_samples_array, ft_times, 'c-+', label="FastText")
    plt.plot(train_samples_array, nb_times, 'r-*', label="NaiveBayes")
    plt.plot(train_samples_array, svm_times, 'g-^', label="SVM")
    plt.plot(train_samples_array, dt_times, 'b-s', label="Decision Tree")
    plt.grid(color='tab:gray', linestyle='-', linewidth=0.15)
    plt.ylabel('time [s]')
    plt.xlabel('number of examples')
    title = 'Total work time - ' + korpus_name
    plt.title(title)
    plt.legend()
    plt.savefig(plot_save_path+title + "." + plotFormat, dpi=dpi, format=plotFormat)
    plt.show()


def draw_accuracy_plot():
    plt.plot(train_samples_array, ft_accuracies, 'c-+', label="FastText")
    plt.plot(train_samples_array, nb_accuracies, 'r-*', label="NaiveBayes")
    plt.plot(train_samples_array, svm_accuracies, 'g-^', label="SVM")
    plt.plot(train_samples_array, dt_accuracies, 'b-s', label="Decision Tree")
    plt.grid(color='tab:gray', linestyle='-', linewidth=0.15)
    plt.ylabel('accuracy')
    plt.xlabel('number of examples')
    title = 'Accuracy - ' + korpus_name
    plt.title(title)
    plt.legend()
    plt.savefig(plot_save_path+title + "." + plotFormat, dpi=dpi, format=plotFormat)
    plt.show()


def draw_roc_auc_score_plot():
    plt.plot(train_samples_array, ft_roc_auc_score_array, 'c-+', label="FastText")
    plt.plot(train_samples_array, nb_roc_auc_score_array, 'r-*', label="NaiveBayes")
    plt.plot(train_samples_array, svm_roc_auc_score_array, 'g-^', label="SVM")
    plt.plot(train_samples_array, dt_roc_auc_score_array, 'b-s', label="Decision Tree")
    plt.grid(color='tab:gray', linestyle='-', linewidth=0.15)
    plt.ylabel('ROC AUC score')
    plt.xlabel('number of examples')
    title = 'ROC AUC score - ' + korpus_name
    plt.title(title)
    plt.legend()
    plt.savefig(plot_save_path+title + "." + plotFormat, dpi=dpi, format=plotFormat)
    plt.show()


def draw_fastText_plot():
    plt.plot(train_samples_array, fastText_accuracies_1, 'r-*', label="epoch = 5")
    plt.plot(train_samples_array, fastText_accuracies_2, 'g-^', label="epoch = 50")
    plt.plot(train_samples_array, fastText_accuracies_3, 'b-s', label="epoch = 200")
    plt.plot(train_samples_array, fastText_accuracies_4, 'm-h', label="epoch = 500")
    plt.plot(train_samples_array, fastText_accuracies_5, 'c-+', label="epoch = 1000")
    plt.grid(color='tab:gray', linestyle='-', linewidth=0.15)
    plt.ylabel('accuracy')
    plt.xlabel('number of examples')
    title = 'Learning curve - fastText'
    plt.title(title)
    plt.legend()
    plt.savefig(plot_save_path+title + "." + plotFormat, dpi=dpi, format=plotFormat)
    plt.show()

step = 0  # do obliczania % ukonczenia

train_samples_array = []

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

files_data = load_files(lemma_dir)

# zamiana bajtów na string
string_data = []
for byteData in files_data.data:
    text = byteData.decode("utf-8")
    string_data.append(text)

svm_clf = LinearSVC()
dt_clf = DecisionTreeClassifier(max_depth=10)
nb_clf = MultinomialNB()

ft_clf = FastText(dim=7, min_count=15, loss='ns', epoch=200, bucket=200000, word_ngrams=1)


for train_size in train_sizes:
    X_train, X_test, y_train, y_test = train_test_split(
        string_data,
        files_data.target,
        train_size=train_size,
        test_size=1 - train_size)

    test_samples_count = len(X_test)
    samples_count = len(X_train)
    print("samples:", samples_count)

    train_samples_array.append(samples_count)

    #learning curve
    ft_roc_auc_score, ft_acc, ft_fit_time, ft_predict_time = fastTextMethod.invoke(
        X_train, X_test, y_train, y_test, ft_clf, iterations)

    nb_roc_auc_score, nb_acc, nb_fit_time, nb_predict_time = bowMethod.invoke(
        X_train, X_test, y_train, y_test, nb_clf, iterations)

    svm_roc_auc_score, svm_acc, svm_fit_time, svm_predict_time = bowMethod.invoke(
        X_train, X_test, y_train, y_test, svm_clf, iterations)

    dt_roc_auc_score, dt_acc, dt_fit_time, dt_predict_time = bowMethod.invoke(
        X_train, X_test, y_train, y_test, dt_clf, iterations)

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

    ft_accuracies.append(ft_acc)
    nb_accuracies.append(nb_acc)
    svm_accuracies.append(svm_acc)
    dt_accuracies.append(dt_acc)

    ft_roc_auc_score_array.append(ft_roc_auc_score)
    nb_roc_auc_score_array.append(nb_roc_auc_score)
    svm_roc_auc_score_array.append(svm_roc_auc_score)
    dt_roc_auc_score_array.append(dt_roc_auc_score)

    # draw_roc_auc_score_plot()
    # draw_accuracy_plot()
    # draw_fit_time_plot()
    # draw_predict_time_plot()
    # draw_time_plot()

    # confusion matrix
    y_pred_ft, fit_time, predict_time, y_score = fastTextMethod.iter_step(X_train, X_test, y_train, ft_clf)
    show_confusion_matrix(y_test, y_pred_ft, files_data.target_names, 'fastText')
    y_pred_nb, fit_time, predict_time, y_score = bowMethod.iter_step(X_train, X_test, y_train, nb_clf)
    show_confusion_matrix(y_test, y_pred_nb, files_data.target_names, 'NaiveBayes')

    y_pred_svm, fit_time, predict_time, y_score = bowMethod.iter_step(X_train, X_test, y_train, svm_clf)
    show_confusion_matrix(y_test, y_pred_svm, files_data.target_names, 'SVM')

    y_pred_dt, fit_time, predict_time, y_score = bowMethod.iter_step(X_train, X_test, y_train, dt_clf)
    show_confusion_matrix(y_test, y_pred_dt, files_data.target_names, 'DecisionTree')

    step += 1
    print("Finished:", str(step / len(train_sizes)) + "%")

