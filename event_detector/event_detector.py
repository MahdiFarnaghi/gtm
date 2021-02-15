import sys
import os
from time import sleep, time
from numpy.lib.type_check import asscalar

from sqlalchemy.sql.operators import exists
from gttm.ts.task_scheduler import TaskScheduler
from gttm.db import PostgresHandler_EventDetection, PostgresHandler_Tweets
from dotenv import load_dotenv
from datetime import datetime, timedelta

from gttm.nlp import VectorizerUtil_FastText
from gttm.ioie.geodata import add_geometry, get_wgs84_crs
from gttm.nlp.identify_topic import HDPTopicIdentification
import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.preprocessing import StandardScaler
import math
import traceback
import nltk

load_dotenv()

db_hostname = os.getenv('DB_HOSTNAME')
db_port = os.getenv('DB_PORT')
db_user = os.getenv('DB_USER')
db_pass = os.getenv('DB_PASS')
db_database = os.getenv('DB_DATABASE')

postgres_tweets = PostgresHandler_Tweets(
    db_hostname, db_port, db_database, db_user, db_pass)
postgres_tweets.check_db()

postgres_events = PostgresHandler_EventDetection(
    db_hostname, db_port, db_database, db_user, db_pass)
postgres_events.check_db()

print('LOADING LANGUAGE MODELS')
languages = os.getenv('LANGUAGES')
if languages != '':
    languages = str(languages).split(',')
    languages = [str.strip(lang) for lang in languages]

vectorizer = VectorizerUtil_FastText()
[vectorizer.get_model(lang) for lang in languages]
print('LOADING LANGUAGE MODELS was finished.')

print('LOADING NLTK')
nltk.download('wordnet', '/data/nltk')
print('LOADING NLTK was finished!')


class EventDetector:
    def __init__(self, check_database_threshold=60):
        """
        Initialize an EventDetector object
        """
        self.check_database_threshold = check_database_threshold

        self.db_hostname = os.getenv('DB_HOSTNAME')
        self.db_port = os.getenv('DB_PORT')
        self.db_user = os.getenv('DB_USER')
        self.db_pass = os.getenv('DB_PASS')
        self.db_database = os.getenv('DB_DATABASE')
        # self._postgres = PostgresHandler_EventDetection('localhost', 5432, 'tweetsdb', 'postgres', 'postgres')
        self.postgres = PostgresHandler_EventDetection(
            self.db_hostname, self.db_port,  self.db_database, self.db_user, self.db_pass)

        self.task_list = {}

        self.scheduler = TaskScheduler()
        self.scheduler.start_scheduler()

    def updates_event_detection_task(self):
        db_tasks = self.postgres.get_tasks()
        for task in db_tasks:
            sch_task = self.scheduler.get_task(str(task['task_id']))
            if sch_task is None:
                if task['lang_code'] in languages:
                    self.scheduler.add_task(execute_event_detection_procedure, interval_minutes=task['interval_min'], args=(
                        task['task_id'], task['task_name'], task['min_x'], task['min_y'], task['max_x'], task['max_y'], task['look_back_hrs'], task['lang_code'],), task_id=task['task_id'])
                else:
                    print(
                        f"The selected language ({task['lang_code']}) is not supported.")

        running_tasks_ids = self.scheduler.get_tasks_ids()
        for task_id in running_tasks_ids:
            if not any(str(task['task_id']) == task_id for task in db_tasks):
                self.scheduler.remove_task(task_id)

        pass

    def run(self):
        while (True):
            self.updates_event_detection_task()
            # Check for new instruction in the database
            sleep(self.check_database_threshold)

exec_number = 0
def execute_event_detection_procedure(task_id: int, task_name: str, min_x, min_y, max_x, max_y, look_back_hours: int, lang_code,
                                      min_cluster_size=10, st_clustering_max_eps=2, text_clustering_max_eps=0.4, verbose=True):

    global postgres_tweets, postgres_events, vectorizer, languages, exec_number
    exec_number = exec_number + 1

    end_date = datetime.now()
    start_date = end_date - timedelta(hours=int(look_back_hours))

    print("*"*60)
    print("*"*60)
    print(
        F"Process: {task_name} ({task_id}), Language: {lang_code}, Interval: {start_date} to {end_date}")
    print(F"Execution number: {exec_number}")

    if not lang_code in languages:
        print(f"The selected language ({lang_code}) is not supported.")
        print('Processing was terminated.')

    # Read data from database
    print("1. Read data from database.")
    df, num = postgres_tweets.read_data_from_postgres(
        start_date=start_date,
        end_date=end_date,
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        lang=lang_code)

    if num <= 0:
        print('There was no record for processing.')
        print('Processing was terminated.')
        return
    if verbose:
        print(F"Number of retrieved tweets: {num}")

    # convert to geodataframe
    print("2. convert to GeoDataFrame")
    gdf = add_geometry(df, crs=get_wgs84_crs())

    # get location vectors
    print("3. Tweet info")
    x = np.asarray(gdf.geometry.x)[:, np.newaxis]
    y = np.asarray(gdf.geometry.y)[:, np.newaxis]
    # get time vector
    t = np.asarray(gdf.created_at.dt.year * 365.2425 + gdf.created_at.dt.day)
    date_time = gdf.created_at.dt.to_pydatetime()
    # get tweet_id and user_id
    tweet_id = gdf.id.values
    user_id = gdf.user_id.values

    # Vectorzie text
    print("4. Get text vector")
    clean_text = df.c.values
    text = df.text.values
    text_vect = None
    text_vect = vectorizer.vectorize(df.c.values, lang_code)
    # Added to debugging
    # if __debug__:
    #     text_vect_path = '~/temp/text.npy'
    #     os.makedirs('~/temp', exist_ok=True)
    #     if os.path.exists(text_vect_path):
    #         text_vect = np.load(text_vect_path)
    #     else:
    #         text_vect = vectorizer.vectorize(df.c.values, lang_code)
    #         np.save(text_vect_path, text_vect)
    # else:
    #     text_vect = vectorizer.vectorize(df.c.values, lang_code)

    # print(F"Shape of the vectorized tweets: {text_vect.shape}")

    # Text-based clustering
    print("5. Clustering - First-level: Text-based")
    start_time = time()
    optics_ = OPTICS(
        min_cluster_size=min_cluster_size,
        max_eps=text_clustering_max_eps,
        metric='precomputed')
    text_dist = np.absolute(cosine_distances(text_vect))
    optics_.fit(text_dist)
    time_taken = time() - start_time
    txt_clust_labels = optics_.labels_
    txt_clust_label_codes = np.unique(txt_clust_labels)
    num_of_clusters = len(txt_clust_label_codes[txt_clust_label_codes >= 0])
    if verbose:
        print(F'\tNumber of clusters: {num_of_clusters - 1}')
        print(F"\tTime: {math.ceil(time_taken)} seconds")
    if num_of_clusters <= 0:
        print("No first level cluster was detected.")
        print('Processing was terminated.')
        return

    # topic identification
    print("6. Identify topics")
    # TODO: We need to specify the maximum number of tweets enter into the clustering procedures
    identTopic = HDPTopicIdentification()
    identTopic.identify_topics(txt_clust_labels, clean_text)
    if verbose:
        identTopic.print_cluster_topics('\t')
    topics = identTopic.get_cluster_topics()

    clusters = []
    print("\n7. Clustering - Second-level: Spatiotemporal")
    for label in txt_clust_label_codes:
        if label >= 0:
            start_time = time()
            optics_ = OPTICS(
                min_cluster_size=min_cluster_size,
                max_eps=st_clustering_max_eps,
                metric='precomputed')
            _x = x[txt_clust_labels == label]
            _y = y[txt_clust_labels == label]
            _tweet_id = tweet_id[txt_clust_labels == label]
            _user_id = user_id[txt_clust_labels == label]
            # _x = StandardScaler().fit_transform(x[txt_clust_labels == label])
            # _y = StandardScaler().fit_transform(y[txt_clust_labels == label])
            _text = text[txt_clust_labels == label]
            _date_time = date_time[txt_clust_labels == label]
            # TODO: How to deal with tweets from a single user?
            st_vect = np.concatenate((_x,
                                      _y,
                                      #   t[txt_clust_labels==label],
                                      ), axis=1)
            st_dist = euclidean_distances(st_vect)
            optics_.fit(st_dist)
            time_taken = time() - start_time
            st_clust_labels = optics_.labels_
            st_clust_label_codes = np.unique(st_clust_labels)
            num_of_clusters = len(
                st_clust_label_codes[st_clust_label_codes >= 0])
            st_any_clust = num_of_clusters > 0
            if verbose:
                print(f'\t{label}: {topics[label][4]}')
                print(
                    F"\t#tweets: {len(st_clust_labels)}, #clusters {num_of_clusters}, time: {math.ceil(time_taken)} seconds")

            for l in st_clust_label_codes[st_clust_label_codes >= 0]:
                topic = topics[label][3]
                topic_words = topics[label][4]
                points_text = _text[st_clust_labels == l].tolist()
                points_x = _x[st_clust_labels == l]
                points_y = _y[st_clust_labels == l]
                points_tweet_id = _tweet_id[st_clust_labels == l]
                points_user_id = _user_id[st_clust_labels == l]
                points_date_time = _date_time[st_clust_labels == l].tolist()
                lat_min = np.min(points_y)
                lat_max = np.max(points_y)
                lon_min = np.min(points_x)
                lon_max = np.max(points_x)
                dt_min = min(points_date_time)
                dt_max = max(points_date_time)
                if (len(np.unique(points_user_id)) > 1):
                    clusters.append({
                        'id': None,
                        'task_id': task_id,
                        'task_name': task_name,
                        'topic': topic,
                        'topic_words': topic_words,
                        'latitude_min': lat_min,
                        'latitude_max': lat_max,
                        'longitude_min': lon_min,
                        'longitude_max': lon_max,
                        'date_time_min': dt_min,
                        'date_time_max': dt_max,
                        'points': [{'cluster_id': None,
                                    'longitude': xx.item(),
                                    'latitude': yy.item(),
                                    'text': tt,
                                    'date_time': dd,
                                    'tweet_id': ti.item(),
                                    'user_id': ui.item()} for xx, yy, tt, dd, ti, ui in zip(points_x, points_y, points_text, points_date_time, points_tweet_id, points_user_id)]
                    })

    print("8. Link clusters")
    # TODO: 8. Link clusters
    # TODO: 8.1 Select cluster that coincide with the current time interval and extent
    # TODO: 8.2 Retrieve their points
    # TODO: 8.3 Compare the point of the old clusters and the new clusters
    # TODO: 8.4 Link the clusters with higher cluster relation strength
    print("9. Save clusters")
    postgres_events.insert_clusters(clusters)

    print(F"Process {task_name} ({task_id}) finished.")
    print('*'*60)
    print("*"*60)


while True:
    try:
        event_detector = EventDetector()
        event_detector.run()        
    except:
        print("&"*60)
        print("&"*60)
        print("Unhandled error caught")
        traceback.print_exc()
        print("&"*60)
        print("&"*60)
