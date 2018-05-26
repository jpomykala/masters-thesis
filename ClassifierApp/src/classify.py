from collections import Counter

import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from src.utils import filter_files, get_test_sample, get_train_sample
from src.xml_reader import read_stop_words_list

files_counter = Counter(f.source for f in files)
print('Counter: ' + str(files_counter))

# twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

articles_count = 500

politics = filter_files("polityka", files)[0:articles_count]
sport = filter_files("sport", files)[0:articles_count]
health = filter_files("zdrowie", files)[0:articles_count]
security = filter_files("bezpieczeństwo", files)[0:articles_count]
history = filter_files("historia", files)[0:articles_count]
hardware = filter_files("sprzęt", files)[0:articles_count]

split = 0.3
stop_words_list = read_stop_words_list("../data/stop_words/list.txt")

test_politics = get_test_sample(split, politics)
test_sport = get_test_sample(split, sport)
test_health = get_test_sample(split, health)
test_hardware = get_test_sample(split, hardware)
test_history = get_test_sample(split, history)
test_security = get_test_sample(split, security)

test_set = test_politics + test_sport + test_health + test_hardware + test_security + test_history
test_counter = Counter(f.category for f in test_set)
print('Test set counter: ' + str(test_counter))
print('Test set files count: ' + str(len(test_set)))

train_politics = get_train_sample(split, politics)
train_sport = get_train_sample(split, sport)
train_health = get_train_sample(split, health)
train_hardware = get_train_sample(split, hardware)
train_history = get_train_sample(split, history)
train_security = get_train_sample(split, security)

train_set = train_politics + train_sport + train_health + train_hardware + train_history + train_security
train_counter = Counter(f.category for f in train_set)
print('Train set: ' + str(train_counter))
print('Train set files count: ' + str(len(train_set)))

le = preprocessing.LabelEncoder()
le.fit([f.category for f in train_set])
print('Classes: ' + str(le.classes_))

train_data = [f.body for f in train_set]
train_target = le.transform([f.category for f in train_set])

test_data = [f.body for f in test_set]
test_target = le.transform([f.category for f in test_set])

count_vectorizer = CountVectorizer(stop_words=stop_words_list)
train_counts = count_vectorizer.fit_transform(train_data)
tfidf_transformer = TfidfTransformer(use_idf=False)
train_tfidf = tfidf_transformer.fit_transform(train_counts)
clf = MultinomialNB()
clf.fit(train_tfidf, train_target)

X_new_counts = count_vectorizer.transform(test_data)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predictedNB = clf.predict(X_new_tfidf)
testNB = np.mean(predictedNB == test_target)
print('NB: ' + str(testNB))
print(metrics.classification_report(test_target, predictedNB, target_names=le.classes_))

# pipelineNB = Pipeline([
#     ('vect', CountVectorizer(stop_words=stop_words_list)),
#     ('tfidf', TfidfTransformer()),
#     ('clf', naive_bayes.MultinomialNB())])
#
# pipelineNB = pipelineNB.fit(train_data, train_target)
# predicted = pipelineNB.predict(test_data)
# testNB = np.mean(predicted == test_target)
# print('NB: ' + str(testNB))


# result = pipeline.predict([politics_article, it_article, health_article])
# print(list(le.inverse_transform(result)))
