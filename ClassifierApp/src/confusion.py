import itertools

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, classes, title, cmap=plt.cm.Blues):
    classes_count = len(classes)
    small_classes = classes_count < 10

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))

    font_size = 5
    rotation = 90
    if small_classes:
        font_size = 8
        rotation = 45

    plt.xticks(tick_marks, classes, rotation=rotation, fontsize=font_size)
    plt.yticks(tick_marks, classes, fontsize=font_size)

    if small_classes:
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Rzeczywista klasa')
    plt.xlabel('Dopasowana klasa ')
