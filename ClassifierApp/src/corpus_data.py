import os
import pathlib
import shutil

import glob2
import matplotlib.pyplot as plt
import numpy as np
from string import digits


def show_summary_plot(path, filter_less_than=0):
    path = end_path_with_slash(path)
    corpus_summary = summarize(path)
    corpus_summary_filtered = {k: v for k, v in corpus_summary.items() if v > filter_less_than}
    plt.gcf().subplots_adjust(left=0.24)
    labels = tuple(corpus_summary_filtered.keys())
    y_pos = np.arange(len(labels))
    values = list(corpus_summary_filtered.values())
    plt.barh(y_pos, values, align='center', alpha=0.5)
    plt.yticks(y_pos, labels)
    plt.xlabel('Liczba dokumentów')
    # plt.savefig('test', dpi=1600)
    plt.show()


def end_path_with_slash(path):
    if not path.endswith("/"):
        path += "/"
    return path


def summarize_plot_words(path):
    path = end_path_with_slash(path)
    output = {}
    for root, dirs, files in os.walk(path):
        for directory in dirs:
            files_count = summarize_words(root + directory)
            output[directory] = files_count

    corpus_summary_filtered = {k: v for k, v in output.items() if v > 0}
    plt.gcf().subplots_adjust(left=0.24)
    labels = tuple(corpus_summary_filtered.keys())
    y_pos = np.arange(len(labels))
    values = list(corpus_summary_filtered.values())
    plt.barh(y_pos, values, align='center', alpha=0.5)
    plt.yticks(y_pos, labels)
    plt.xlabel('Średnia ilość lematów')
    plt.show()
    return output


def summarize_words(category_dir):
    category_dir = end_path_with_slash(category_dir)
    all_in_path = glob2.glob(category_dir + '/*.*')
    count = 0
    for path in all_in_path:
        with open(path, "rb") as file:
            utf_text = file.read().decode('utf-8')
            words_count = len(utf_text.split())
            count += words_count
    return count / len(all_in_path)


def summarize(category_dir):
    output = {}
    for root, dirs, files in os.walk(category_dir):
        for directory in dirs:
            files_count = sum_files(root + directory)
            output[directory] = files_count
    return output


def sum_files(directory):
    files_list = os.listdir(directory)
    return len(files_list)


def redistribute_to_categories(files, dir_path, min_per_category=0, max_per_category=100):
    dir_path = end_path_with_slash(dir_path)
    shutil.rmtree(dir_path, ignore_errors=True)
    category_count_dict = {}
    for file in files:
        source = file.source
        category = file.category

        if "zaufanatrzeciastrona.pl" in source:
            category = "bezpieczeństwo"

        if "niebezpiecznik.pl" in source:
            category = "bezpieczeństwo"

        if "purepc.pl" in source:
            category = "sprzęt"

        if "pap.pl/aktualnosci/kraj/" in source:
            category = "polityka"

        if "pap.pl/aktualnosci/sport/" in source:
            category = "sport"

        if "kafeteria.pl" in source:
            category = "zdrowie"

        if category not in category_count_dict:
            category_count_dict[category] = 0

        count = category_count_dict[category]
        if count >= max_per_category:
            continue

        new_count = count + 1
        category_count_dict[category] = new_count
        write_to_category(file, dir_path, category)

    for k, v in category_count_dict.items():
        if v < min_per_category:
            print("Remove: " + k + " didn't meet requirements, files:", v)
            shutil.rmtree(dir_path + k, ignore_errors=True)


file_id = 0


def redistribute_to_categories_wiki(from_dir, to_dir):
    from_dir = end_path_with_slash(from_dir)
    shutil.rmtree(to_dir, ignore_errors=True)

    for file_name in os.listdir(from_dir):
        category_name = get_category_name(file_name)
        file = open(from_dir + file_name, 'r')
        body = file.read()
        write_to_category_v2(body, to_dir, category_name)


def get_category_name(file_name):
    remove_digits = str.maketrans('', '', digits)
    output = file_name.translate(remove_digits)
    output = output.replace('_', '')
    output = output.replace('.txt', '')
    return output


def write_to_category_v2(text, path_to_category, category):
    global file_id
    file_id += 1

    path_to_file = path_to_category + "/" + category + "/"
    pathlib.Path(path_to_file).mkdir(parents=True, exist_ok=True)
    with open(path_to_file + str(file_id) + '.txt', "w+", encoding="utf-8") as outfile:
        outfile.write(text)


def write_to_category(file, path_to_category, category):
    global file_id
    file_id += 1

    path_to_file = path_to_category + "/" + category + "/"
    pathlib.Path(path_to_file).mkdir(parents=True, exist_ok=True)
    with open(path_to_file + str(file_id) + '.txt', "w+", encoding="utf-8") as outfile:
        outfile.write(file.body)
