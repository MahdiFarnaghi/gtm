from gttm.db.postgres import PostgresHandler_Tweets
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


def retrieve_data():
    load_dotenv()

    print('Reading the env variables ...')
    save_data_mode = os.getenv('SAVE_DATA_MODE')
    if save_data_mode is None:
        raise Exception('Save data mode was not properly set.')

    # TODO: The boundary should be loaded from the tasks tables and it should be iteratively updated.
    min_x = float(os.getenv('MIN_X')) if os.getenv('MIN_X') is not None else 0
    max_x = float(os.getenv('MAX_X')) if os.getenv('MAX_X') is not None else 0
    min_y = float(os.getenv('MIN_Y')) if os.getenv('MIN_Y') is not None else 0
    max_y = float(os.getenv('MAX_Y')) if os.getenv('MAX_Y') is not None else 0
    area_name = os.getenv('AREA_NAME')
    print(f"Boundary ({area_name}): {min_x}, {min_y}, {max_x}, {max_y}")

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

    languages = os.getenv('LANGUAGES')
    if languages != '':
        languages = str(languages).split(',')
        languages = [str.strip(lang) for lang in languages]
        print(f"Languages: {languages}")
    else:
        print(f"Languages: {languages}")
        raise Exception("Languages were not set properly.")

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
    print('Authenication was successful.')

    print(f"Save data mode: {save_data_mode}")
    postgres = None
    if save_data_mode == 'DB':
        postgres = PostgresHandler_Tweets(
            db_hostname, db_port, db_database, db_user, db_pass)
        print('Checking the database ...')
        try:
            print(F"Check database result: {postgres.check_db()}")
        except:
            print('Database is not accessible!')
            print('-' * 60)
            print("Unexpected error:", sys.exc_info()[0])
            print('-' * 60)
            traceback.print_exc(file=sys.stdout)
            print('-' * 60)
            sys.exit(2)

    api = tweepy.API(auth)
    print(60*"*")
    print(60*"*")
    print(60*"=")
    print(60*"*")

    print('Initializing the listener ...')
    WorldStreamListener = TwitterStreamListener()
    WorldStreamListener.init(area_name=area_name,  output_folder=output_folder,
                             postgres=postgres, save_data_mode=save_data_mode)
    WorldStream = Stream(auth, WorldStreamListener)
    WorldStream.filter(languages=languages if len(languages)
                       > 0 else None, locations=[min_x, min_y, max_x, max_y])


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
