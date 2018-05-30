# from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from src.xml_reader import read_stop_words_list

lemma_dir = "../data/korpus/lemma"
lemma_data = load_files(lemma_dir)

string_data = []
for byteData in lemma_data.data:
    text = byteData.decode("utf-8")
    string_data.append(text)


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % topic_idx)
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))


stop_words_list = read_stop_words_list("../data/stop_words/list.txt")
train_size = 0.7

data_train, data_test, target_train, target_test = train_test_split(
    string_data,
    lemma_data.target,
    train_size=train_size,
    test_size=1 - train_size)

documents = data_train

no_features = 1000

tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words=stop_words_list)
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 7

lda = LatentDirichletAllocation(n_topics=no_topics,
                                max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda.fit(tf)

no_top_words = 10
display_topics(lda, tf_feature_names, no_top_words)


