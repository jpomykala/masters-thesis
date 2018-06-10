import os
import collections
import glob2
from xml.etree import ElementTree
from random import shuffle

from src.consts import stop_words_list_path
from src.utils import save_text


def ccl_to_lemma(input_file, output_file):
    tree = ElementTree.parse(input_file)
    root = tree.getroot()
    base_words = []
    for token in root.getiterator('tok'):
        base = token.find('lex').find('base').text
        base_words.append(base)

    output_file = output_file + ".lemma"
    content = " ".join(base_words)
    save_text(output_file, content)


def ccl_to_lemma_noun(input_file, output_file):
    tree = ElementTree.parse(input_file)
    root = tree.getroot()
    base_words = []
    for token in root.getiterator('tok'):
        base = token.find('lex').find('base').text
        c_tag = token.find('lex').find('ctag').text
        if 'subst' in c_tag:
            base_words.append(base)

    output_file = output_file + ".noun"
    content = " ".join(base_words)
    save_text(output_file, content)


def ccl_to_lemma_noun(input_file, output_file):
    tree = ElementTree.parse(input_file)
    root = tree.getroot()
    base_words = []
    for token in root.getiterator('tok'):
        base = token.find('lex').find('base').text
        c_tag = token.find('lex').find('ctag').text
        if 'subst' in c_tag:
            base_words.append(base)
        if 'subst' in c_tag:
            base_words.append(base)

    output_file = output_file + ".noun"
    content = " ".join(base_words)
    save_text(output_file, content)


def read_stop_words_list():
    output = list(open(stop_words_list_path, encoding="utf-8"))
    return [l.replace('\n', '') for l in output]


def read_files_from_dir(path, max_per_dir):
    directories = glob2.glob(path + "/*/")
    output = []
    for d in directories:
        tmp = read_random_files(d, max_per_dir)
        output += tmp
    return output


def read_random_files(path, max_files_count):
    files_count = 0
    output = []
    all_xml_in_path = glob2.glob(path + '/**/*.xml')
    shuffle(all_xml_in_path)
    for xml_file in all_xml_in_path:
        try:
            parsed_file = read_file(xml_file)
            output.append(parsed_file)
            files_count += 1
        except:
            pass

        if files_count >= max_files_count:
            break

    print(str(len(output)) + " files from " + path + " in scope")
    return output


def read_file(path_to_file):
    TextData = collections.namedtuple('TextData', 'body category source')
    full_path = os.path.abspath(path_to_file)
    dom = ElementTree.parse(full_path)
    body = dom.find('body').text
    category = dom.find('category').text
    source = dom.find('source').text

    if body is None:
        raise Exception("No body")

    if category is None:
        raise Exception('No category')

    if source is None:
        raise Exception('No source')

    return TextData(body=body, category=category.lower(), source=source.lower())


