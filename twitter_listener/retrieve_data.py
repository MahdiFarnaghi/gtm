from listener import TwitterStreamListener
import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import datetime
import os
import json
from dotenv import load_dotenv
from db.postgres import PostgresHandler

load_dotenv()

# Koohnavard account
print('Reading the env variables ...')
save_data_mode = os.getenv('SAVE_DATA_MODE')
consumer_key = os.getenv('CONSUMER_KEY')
consumer_secret = os.getenv('CONSUMER_SECRET')
access_token = os.getenv('ACCESS_TOKEN')
access_secret = os.getenv('ACCESS_SECRET')
output_folder = os.getenv('TWEETS_OUPUT_FOLDER')
min_x = float(os.getenv('MIN_X')) if os.getenv('MIN_X') is not None else 0
max_x = float(os.getenv('MAX_X')) if os.getenv('MAX_X') is not None else 0
min_y = float(os.getenv('MIN_Y')) if os.getenv('MIN_Y') is not None else 0
max_y = float(os.getenv('MAX_Y')) if os.getenv('MAX_Y') is not None else 0
if not (min_x != 0 and max_x != 0 and min_y != 0 and max_y != 0):
    raise Exception("The bounding box for which the tweets should be saved is not provided.")
area_name = os.getenv('AREA_NAME')
db_hostname = os.getenv('DB_HOSTNAME')
db_port = os.getenv('DB_PORT')
db_user = os.getenv('DB_USER')
db_pass = os.getenv('DB_PASS')
db_database = os.getenv('DB_DATABASE')
# print(60*"*")
# print("consumer_key: " + consumer_key)
# print("consumer_secret: " + consumer_secret)
# print("access_token: " + access_token)
# print("access_secret: " + access_secret)
# print("output_folder: " + output_folder)
print(60*"*")
print(F"Host name: {db_hostname}")
print(F"Port: {db_port}")
print(F"User: {db_user}")
print(F"Pass: {db_pass}")
print(F"DB_name: {db_database}")
print(60*"*")

print('Reading the env variables was finished!')

print('Authenication ...')
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
print('Authenication was finished!')

postgres = None
if save_data_mode == 'DB':
    postgres = PostgresHandler(
        db_hostname, db_port, db_database, db_user, db_pass)
    print(60*"*")
    print('Checking the database ...')
    print(F"Database status: {postgres.check_db()}")
    print('Checking the database was finished.')
    print(60*"*")

api = tweepy.API(auth)
print(60*"*")
print(60*"*")
print(60*"=")
print(60*"*")
print(60*"*")
print('Initializing the listener ...')
WorldStreamListener = TwitterStreamListener()
WorldStreamListener.init(area_name=area_name,  output_folder=output_folder, postgres=postgres, save_data_mode=save_data_mode)
WorldStream = Stream(auth, WorldStreamListener)
WorldStream.filter(locations=[min_x, min_y, max_x, max_y])

