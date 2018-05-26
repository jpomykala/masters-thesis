import os
import pathlib

import numpy as np
import pandas as pd


def filter_files(category, files):
    return list(filter(lambda x: category in x.category, files))


def get_test_sample(test_split, data):
    all_data_count = len(data)
    test_data_count = int(np.ceil(test_split * all_data_count))
    return data[0: test_data_count]


def get_train_sample(test_split, data):
    all_data_count = len(data)
    test_data_count = int(np.ceil(test_split * all_data_count))
    return data[test_data_count:all_data_count - 1]


def save_text(path, text):
    path_to_create = os.path.dirname(path)
    pathlib.Path(path_to_create).mkdir(parents=True, exist_ok=True)
    with open(path, "w+", encoding="utf-8") as outfile:
        outfile.write(text)


def report_to_csv(report, name):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split(' ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    data_frame = pd.DataFrame.from_dict(report_data)
    data_frame.to_csv(name + '.csv', index=False)

