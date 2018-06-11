import os
import pathlib
import shutil
from collections import OrderedDict, Counter
from xml.etree import ElementTree

import glob2
import matplotlib.pyplot as plt
import numpy as np
from string import digits

from src.consts import plot_save_path, plotFormat, dpi


def count_files_for_path(file_name, path, filter_less_than=0):
    path = end_path_with_slash(path)
    corpus_summary = summarize(path)
    output = {k: v for k, v in corpus_summary.items() if v > filter_less_than}
    output = OrderedDict(sorted(output.items(), key=lambda kv: kv[1]))
    plt.gcf().subplots_adjust(left=0.24)
    labels = tuple(output.keys())
    y_pos = np.arange(len(labels))
    values = list(output.values())
    plt.barh(y_pos, values, align='center', alpha=0.5)
    plt.yticks(y_pos, labels)
    plt.xlabel('Liczba dokumentów')
    plt.savefig(plot_save_path +
                file_name + "." +
                plotFormat, dpi=dpi,
                format=plotFormat)
    plt.show()


def show_classes_summary_plot(path, file_name, title):
    path = end_path_with_slash(path)

    all_in_path = glob2.glob(path + '/**/*.*')
    output = {}
    for path in all_in_path:
        file_count = count_classes_for_file(path)
        output = {k: output.get(k, 0) + file_count.get(k, 0) for k in set(output) | set(file_count)}

    output = {k: v for k, v in output.items() if v > 30000}

    output = OrderedDict(sorted(output.items(), key=lambda kv: kv[1]))

    ignCount = output['ign']
    allCount = sum(output.values()) - ignCount

    result = ignCount / allCount
    print(result * 100)

    plt.gcf().subplots_adjust(left=0.24)
    labels = tuple(output.keys())
    y_pos = np.arange(len(labels))
    values = list(output.values())
    plt.barh(y_pos, values, align='center', alpha=0.5)
    plt.title(title)
    plt.yticks(y_pos, labels)
    plt.xlabel('Liczba wystąpień')
    plt.savefig(plot_save_path +
                file_name + "." +
                plotFormat, dpi=dpi,
                format=plotFormat)
    plt.show()


def summary_all_ign_words(path, file_name, title, filter=50):
    path = end_path_with_slash(path)

    all_in_path = glob2.glob(path + '/**/*.*')
    output = []
    for path in all_in_path:
        ign_words = get_all_ign(path)
        output = output + ign_words

    output = Counter(output)
    output = {k: v for k, v in output.items() if v > filter}
    output = OrderedDict(sorted(output.items(), key=lambda kv: kv[1]))

    plt.gcf().subplots_adjust(left=0.24)
    labels = tuple(output.keys())
    y_pos = np.arange(len(labels))
    values = list(output.values())
    plt.barh(y_pos, values, align='center', alpha=0.5)
    plt.title(title)
    plt.yticks(y_pos, labels)
    plt.xlabel('Liczba wystąpień')
    plt.savefig(plot_save_path +
                file_name + "." +
                plotFormat, dpi=dpi,
                format=plotFormat)
    plt.show()


def get_all_ign(input_file):
    tree = ElementTree.parse(input_file)
    root = tree.getroot()
    output = []
    for token in root.getiterator('tok'):
        c_tag = token.find('lex').find('ctag').text
        base = token.find('lex').find('base').text
        tags_array = c_tag.split(":")
        for tag in tags_array:
            if 'ign' in tag:
                output.append(base.lower())
    return output


def count_classes_for_file(input_file):
    tree = ElementTree.parse(input_file)
    root = tree.getroot()
    output = {}
    for token in root.getiterator('tok'):
        c_tag = token.find('lex').find('ctag').text
        base = token.find('lex').find('base').text
        tags_array = c_tag.split(":")
        for tag in tags_array:
            if tag in output:
                output[tag] = output[tag] + 1
            else:
                output[tag] = 1

            # if 'ign' in tag:
                # print(base)

    if 'ign' in output:
        output['ign'] = output['ign'] * 2.2
    return output


def end_path_with_slash(path):
    if not path.endswith("/"):
        path += "/"
    return path


def average_words_count(path, file_name):
    path = end_path_with_slash(path)
    output = {}
    for root, dirs, files in os.walk(path):
        for directory in dirs:
            files_count = summarize_words(root + directory)
            output[directory] = files_count

    output = {k: v for k, v in output.items() if v > 0}
    output = OrderedDict(sorted(output.items(), key=lambda kv: kv[1]))
    plt.gcf().subplots_adjust(left=0.24)
    labels = tuple(output.keys())
    y_pos = np.arange(len(labels))
    values = list(output.values())
    plt.barh(y_pos, values, align='center', alpha=0.5)
    plt.yticks(y_pos, labels)
    plt.xlabel('Średnia liczba lematów')
    plt.savefig(plot_save_path +
                file_name + "." +
                plotFormat, dpi=dpi,
                format=plotFormat)
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
