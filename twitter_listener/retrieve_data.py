from gttm.db.postgres import PostgresHandler_Tweets, PostgresHandler_EventDetection
from listener import TwitterStreamListener
import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import datetime
import os
import json
import traceback
import sys
from dotenv import load_dotenv
import traceback
import multiprocessing
import time

import nltk

print('LOADING NLTK')
nltk.download('wordnet', '/data/nltk')
print('LOADING NLTK was finished!')

def retrieve_data():
    load_dotenv()
    
    print(60*"*")
    print(60*"*")    
    print('Start Tweet Listener.')
    print(60*"=")
    print(60*"*")

    save_data_mode = os.getenv('SAVE_DATA_MODE')
    if save_data_mode is None:
        print("save_data_mode was not available.")
        save_data_mode = 'FILE'
    print(f"Save data mode: {save_data_mode}")

    consumer_key = os.getenv('CONSUMER_KEY')
    consumer_secret = os.getenv('CONSUMER_SECRET')
    access_token = os.getenv('ACCESS_TOKEN')
    access_secret = os.getenv('ACCESS_SECRET')
    output_folder = os.getenv('TWEETS_OUPUT_FOLDER')
    if consumer_key is None or consumer_secret is None or access_token is None or access_secret is None:
        print(60*"*")
        print("consumer_key: " + consumer_key)
        print("consumer_secret: " + consumer_secret)
        print("access_token: " + access_token)
        print("access_secret: " + access_secret)
        print("output_folder: " +
              (output_folder if output_folder is not None else ''))
        raise Exception(
            "Twitter authenication information was not provided properly.")

    print('Authenication ...')
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth)
    print('Authenication was successful.')
    
    tweets_for_tasks = os.getenv('TWEETS_FOR_TASKS')
    if tweets_for_tasks is None:
        tweets_for_tasks = False
    elif str(tweets_for_tasks).lower() in ['true', '1']:
        tweets_for_tasks = True
    else:
        tweets_for_tasks = False

    check_db_interval = os.getenv('CHECK_DB_INTERVAL')
    if check_db_interval is None:
        check_db_interval = 60
    else: 
        check_db_interval = float(check_db_interval)

    postgres_tweets = None
    postgres_event_detection = None

    if save_data_mode == 'FILE':
        min_x = float(os.getenv('MIN_X')) if os.getenv(
            'MIN_X') is not None else 0
        max_x = float(os.getenv('MAX_X')) if os.getenv(
            'MAX_X') is not None else 0
        min_y = float(os.getenv('MIN_Y')) if os.getenv(
            'MIN_Y') is not None else 0
        max_y = float(os.getenv('MAX_Y')) if os.getenv(
            'MAX_Y') is not None else 0
        area_name = os.getenv('AREA_NAME')
        if min_x == 0 and min_y == 0 and max_x == 0 and max_y == 0:
            print(f"BBOX ({area_name}): {min_x}, {min_y}, {max_x}, {max_y}")
            raise Exception("BBOX was not set properly")
        
        languages = os.getenv('LANGUAGES')
        if languages != '':
            languages = str(languages).split(',')
            languages = [str.strip(lang) for lang in languages]
            print(f"Languages: {languages}")
        else:
            print(f"Languages: {languages}")
            raise Exception("Languages were not set properly.")

        listen_to_tweets(area_name, output_folder, None,
                        save_data_mode, auth, languages, min_x, min_y, max_x, max_y)
    elif save_data_mode == 'DB':    
        db_hostname = os.getenv('DB_HOSTNAME')
        db_port = os.getenv('DB_PORT')
        db_user = os.getenv('DB_USER')
        db_pass = os.getenv('DB_PASS')
        db_database = os.getenv('DB_DATABASE')
        if db_hostname is None or db_port is None or db_user is None or db_pass is None or db_database is None:
            print(60*"*")
            print(F"Host name: {db_hostname}")
            print(F"Port: {db_port}")
            print(F"User: {db_user}")
            print(F"Pass: {db_pass}")
            print(F"DB_name: {db_database}")
            raise Exception("Database inforation was not provided properly.")
        postgres_tweets = PostgresHandler_Tweets(
            db_hostname, db_port, db_database, db_user, db_pass)
        print('Checking the database ...')

        try:
            print(F"Check database result: {postgres_tweets.check_db()}")
        except:
            print('Database is not accessible!')
            print('-' * 60)
            print("Unexpected error:", sys.exc_info()[0])
            print('-' * 60)
            traceback.print_exc(file=sys.stdout)
            print('-' * 60)
            sys.exit(2)

        if not tweets_for_tasks:
            min_x = float(os.getenv('MIN_X')) if os.getenv(
                'MIN_X') is not None else 0
            max_x = float(os.getenv('MAX_X')) if os.getenv(
                'MAX_X') is not None else 0
            min_y = float(os.getenv('MIN_Y')) if os.getenv(
                'MIN_Y') is not None else 0
            max_y = float(os.getenv('MAX_Y')) if os.getenv(
                'MAX_Y') is not None else 0
            area_name = os.getenv('AREA_NAME')

            if min_x == 0 and min_y == 0 and max_x == 0 and max_y == 0:
                print(f"BBOX ({area_name}): {min_x}, {min_y}, {max_x}, {max_y}")
                raise Exception("BBOX was not set properly")

            languages = os.getenv('LANGUAGES')
            if languages != '':
                languages = str(languages).split(',')
                languages = [str.strip(lang) for lang in languages]
                print(f"Languages: {languages}")
            else:
                print(f"Languages: {languages}")
                raise Exception("Languages were not set properly.")

            listen_to_tweets(area_name, output_folder, postgres_tweets,
                             save_data_mode, auth, languages, min_x, min_y, max_x, max_y)
        else:
            postgres_event_detection = PostgresHandler_EventDetection(
                db_hostname, db_port, db_database, db_user, db_pass)
            postgres_event_detection.check_db()

            process = None
            min_x = max_x = min_y = max_y = None
            languages = None
            while True:
                if process is None:
                    min_x, min_y, max_x,  max_y = postgres_event_detection.get_tasks_bbox()
                    languages = postgres_event_detection.get_tasks_languages()                    
                    if min_x is not None and max_x is not None and min_y is not None and max_y is not None and len(languages) > 0:
                        process = multiprocessing.Process(target=listen_to_tweets, args=(
                            'Tasks', output_folder, postgres_tweets, save_data_mode, auth, languages, min_x, min_y, max_x, max_y,))
                        process.start()
                else:
                    _min_x, _min_y, _max_x, _max_y = postgres_event_detection.get_tasks_bbox()
                    _languages = postgres_event_detection.get_tasks_languages()                    

                    if min_x is not None and max_x is not None and min_y is not None and max_y is not None and languages is not None:
                        def Diff(li1, li2):
                            return (list(list(set(li1)-set(li2)) + list(set(li2)-set(li1))))

                        if min_x != _min_x or max_x != _max_x or min_y != _min_y or max_y != _max_y or len(Diff(_languages, languages)) > 0:
                            print("Restarting the process due to change in the event detection tasks.")
                            process.terminate()
                            process = None
                            min_x = _min_x
                            max_x = _max_x
                            min_y = _min_y
                            max_y = _max_y
                            languages = _languages.copy()
                            if min_x is not None and max_x is not None and min_y is not None and max_y is not None and len(languages) > 0:
                                process = multiprocessing.Process(target=listen_to_tweets, args=(
                                    'Tasks', output_folder, postgres_tweets, save_data_mode, auth, languages, min_x, min_y, max_x, max_y,))
                                process.start()
                time.sleep(check_db_interval)
    # listen_to_tweets(area_name, output_folder, postgres, save_data_mode, auth, languages, min_x, min_y, max_x, max_y)


def listen_to_tweets(area_name, output_folder, postgres, save_data_mode, auth, languages: list, min_x, min_y, max_x, max_y):
    print(60*"*")

    print('Initializing the listener ...')
    print(f"BBOX ({area_name}): {min_x}, {min_y}, {max_x}, {max_y}")
    print(f"Languages: {', '.join(languages)}")
    WorldStreamListener = TwitterStreamListener()
    WorldStreamListener.init(area_name=area_name,  output_folder=output_folder,
                             postgres=postgres, save_data_mode=save_data_mode)
    WorldStream = Stream(auth, WorldStreamListener)
    if len(languages) > 0:
        WorldStream.filter(languages=languages, locations=[min_x, min_y, max_x, max_y])
    else:
        WorldStream.filter(locations=[min_x, min_y, max_x, max_y])


while True:
    try:
        retrieve_data()
    except:
        print("&"*60)
        print("&"*60)
        print("Unhandled error caught")
        traceback.print_exc()
        print("&"*60)
        print("&"*60)
