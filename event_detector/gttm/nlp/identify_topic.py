import multiprocessing
import warnings
from abc import abstractmethod
from datetime import datetime
import os
from pathlib import Path

from gensim import corpora, models
import numpy as np
import csv
import re
# from gttm.word_cloud.wordcloud import generate_wordcloud_of_topic


class TopicIdentifier:
    def __init__(self):
        self.cleaned = []
        self.all_tweets_of_clusters = []
        self._model = None
        self.label_codes = None
        self.corpus = None
        self.num_topics = 20
        self.num_words = 10
        self.topics = []
        self.topics_not_formatted = []
        pass

    def get_cluster_topics(self):
        cluster_topics = {}
        if len(self.topics) <= 0:
            return cluster_topics
        if self._model is not None and self.label_codes is not None:
            t_idx = 0
            for idx, l_code in enumerate(self.label_codes):
                if l_code != -1:
                    topics_of_cluster = self._model[self.corpus[t_idx]]
                    best_topic = max(topics_of_cluster,
                                     key=lambda item: item[1])
                    try:
                        topic = self.topics[best_topic[0]]
                        topic_words = re.sub(r'((?P<brackets>[()])|(?P<number>\-?\d*\.?\d+)|(?P<operator>[+\-\*\/]))',
                                             ' ', topic)
                        topic_words = '{}'.format(
                            ' '.join(map(str.strip, topic_words.split())))
                        cluster_topics[l_code] = [
                            l_code, best_topic[0], best_topic[1], topic, topic_words]
                    except:
                        print('Error')
                    t_idx = t_idx + 1
        return cluster_topics

    def get_cluster_topics_not_fromated(self):
        cluster_topics = {}
        if len(self.topics) <= 0:
            return cluster_topics
        if self._model is not None and self.label_codes is not None:
            t_idx = 0
            for idx, l_code in enumerate(self.label_codes):
                if l_code != -1:
                    topics_of_cluster = self._model[self.corpus[t_idx]]
                    best_topic = max(topics_of_cluster,
                                     key=lambda item: item[1])
                    try:
                        cluster_topics[l_code] = [
                            l_code, best_topic[0], best_topic[1], self.topics_not_formatted[best_topic[0]]]
                    except:
                        print('Error')
                    t_idx = t_idx + 1
        return cluster_topics

    def save_cluster_topics(self, folder_path, prefix):
        cluster_topics = self.get_cluster_topics()
        topics_file_path = folder_path + os.sep + \
            "{}_cluster_topics.csv".format(prefix)
        # relevance_file_path = folder_path + os.sep + "{}_cluster_topics_relevance.csv".format(prefix)

        if folder_path != "" and prefix != "":
            if not Path(topics_file_path).parent.exists():
                os.makedirs(Path(topics_file_path).parent.absolute())
            with open(topics_file_path, 'w', newline='', encoding='utf-8') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerows(
                    [['Cluster Label', 'Topic Code', 'Topic relevance (percent)', 'Topic', 'Topic words']])
                writer.writerows(cluster_topics)

    # def save_wordcloud_of_cluster_topics(self, folder_path, prefix, label_code_for_plotting=None, multiprocess=False,
    #                                      verbose=False):
    #     if verbose:
    #         print('\tStart creating wordcloud of topics ...')
    #     s_time = datetime.now()
    #     with warnings.catch_warnings():
    #         cluster_topics = self.get_cluster_topics_not_fromated()
    #         proc_lst = []
    #         for t in cluster_topics:
    #             plot_cluster_topic = label_code_for_plotting is None or (
    #                     label_code_for_plotting is not None and t[0] in label_code_for_plotting)
    #             if plot_cluster_topic:
    #                 topic = t[3]
    #                 topic_words = dict(topic)
    #                 file_path = folder_path + os.sep + 'cluster_topic_wordcloud' + os.sep + '{}_cluster{:02d}_topic_wordcloud.png'.format(
    #                     prefix, int(t[0]))
    #                 if multiprocess:
    #                     p = multiprocessing.Process(target=generate_wordcloud_of_topic,
    #                                                 args=(topic_words, True, file_path,))
    #                     proc_lst.append(p)
    #                     p.start()
    #                 else:
    #                     generate_wordcloud_of_topic(topic_words, True, file_path)
    #                 # generate_wordcloud_of_topic(topic_words, None, True, file_path)
    #         if multiprocess:
    #             for p in proc_lst:
    #                 p.join()

    #     dur = datetime.now() - s_time
    #     if verbose:
    #         print('\tCreating wordcloud of topics was finished ({} seconds).'.format(dur.seconds))
    #     pass

    def print_cluster_topics(self, prefix=''):
        cluster_topics = self.get_cluster_topics()
        if len(cluster_topics) <= 0:
            print("\tNo topic identified!")
            return
        print("\tTopics")
        for key in cluster_topics:
            topic = cluster_topics[key]
            print(f"{prefix}{key}: {topic[4]}")
            # print("\t\tCluster {}: (Topic code {}, Relevance {}) Topic: {}".format(topic[0], topic[1], topic[2],
            #                                                                        topic[3]))
        # if self._model is not None and self.label_codes is not None:
        #     t_idx = 0
        #     for idx, l_code in enumerate(self.label_codes):
        #         if l_code != -1:
        #             topics_of_cluster = self._model[self.corpus[t_idx]]
        #             best_topic = max(topics_of_cluster, key=lambda item: item[1])
        #             print("\t\tCluster {}: {}".format(l_code, best_topic))
        #             print("\t\t\tTopic: {}".format(self.topics[best_topic[0]]))
        #             t_idx = t_idx + 1

    def print_topics(self):
        if self._model is not None:
            for i, topic in self._model.show_topics(formatted=True, num_topics=self.num_topics,
                                                    num_words=self.num_words):
                print("\t\t" + str(i) + ": " + topic)
                print()

    @abstractmethod
    def identify_topics(self, labels, texts):
        pass


# a good reference: https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21
# another good reference: https://towardsdatascience.com/the-complete-guide-for-topics-extraction-in-python-a6aaa6cedbbc
class LDATopicIndentifier(TopicIdentifier):
    def __init__(self):
        TopicIdentifier.__init__(self)
        pass

    def identify_topics(self, labels, texts):
        self.label_codes = np.unique(labels)
        for idx, l_code in enumerate(self.label_codes):
            if l_code != -1:
                all_tweets_of_cluster = " ".join(texts[labels == l_code])
                self.all_tweets_of_clusters.append(all_tweets_of_cluster)
                self.cleaned.append(all_tweets_of_cluster.split(' '))

        self.num_topics = len(self.label_codes[self.label_codes != -1]) * 2
        dictionary = corpora.Dictionary(self.cleaned)
        self.corpus = [dictionary.doc2bow(cleandoc)
                       for cleandoc in self.cleaned]
        self._model = models.ldamodel.LdaModel(
            self.corpus, num_topics=self.num_topics, id2word=dictionary)
        if self._model is not None:
            for i, topic in self._model.show_topics(formatted=True, num_topics=self.num_topics, num_words=10):
                self.topics.append(topic)

            for i, topic in self._model.show_topics(formatted=False,
                                                    num_topics=self.num_topics,
                                                    num_words=10):
                self.topics_not_formatted.append(topic)

        if len(self.topics) < 5:
            print()


# a good reference: https://radimrehurek.com/gensim/models/hdpmodel.html
class HDPTopicIdentification(TopicIdentifier):
    def __init__(self):
        TopicIdentifier.__init__(self)
        pass

    def identify_topics(self, labels, texts, verbose=False):
        if verbose:
            print('\tStart identifying topics ...')
        s_time = datetime.now()

        self.label_codes = np.unique(labels)
        for idx, l_code in enumerate(self.label_codes):
            if l_code != -1:
                all_tweets_of_cluster = " ".join(texts[labels == l_code])
                self.all_tweets_of_clusters.append(all_tweets_of_cluster)
                self.cleaned.append(all_tweets_of_cluster.split(' '))

        dictionary = corpora.Dictionary(self.cleaned)
        self.corpus = [dictionary.doc2bow(cleandoc)
                       for cleandoc in self.cleaned]
        self._model = models.HdpModel(self.corpus, dictionary)
        self.num_topics = self._model.get_topics().shape[0]
        if self._model is not None:
            for i, topic in self._model.show_topics(formatted=True,
                                                    num_topics=self.num_topics,
                                                    num_words=10):
                self.topics.append(topic)

            for i, topic in self._model.show_topics(formatted=False,
                                                    num_topics=self.num_topics,
                                                    num_words=10):
                self.topics_not_formatted.append(topic)

        if len(self.topics) < 5:
            print()
        dur = datetime.now() - s_time
        if verbose:
            print('\tIdentifying topics was finished ({} seconds).'.format(dur.seconds))
        pass
