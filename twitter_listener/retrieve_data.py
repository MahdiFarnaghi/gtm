from listener import TwitterStreamListener
import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import datetime
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Koohnavard account
print('Reading the env variables ...')
consumer_key = os.getenv('CONSUMER_KEY')
consumer_secret = os.getenv('CONSUMER_SECRET')
access_token = os.getenv('ACCESS_TOKEN')
access_secret = os.getenv('ACCESS_SECRET')
output_folder = os.getenv('TWEETS_OUPUT_FOLDER')
min_x = float(os.getenv('MIN_X'))
max_x = float(os.getenv('MAX_X'))
min_y = float(os.getenv('MIN_Y'))
max_y = float(os.getenv('MAX_Y'))
area_name = os.getenv('AREA_NAME')

# print(60*"*")
# print("consumer_key: " + consumer_key)
# print("consumer_secret: " + consumer_secret)
# print("access_token: " + access_token)
# print("access_secret: " + access_secret)
# print("output_folder: " + output_folder)
# print(60*"*")        
print('Reading the env variables was finished!')

print('Authenication ...')
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
print('Authenication was finished!')

api = tweepy.API(auth)

print('Initializing the listener ...')
WorldStreamListener = TwitterStreamListener()
WorldStreamListener.init(area_name, output_folder)
WorldStream = Stream(auth, WorldStreamListener)
# WorldStream.filter(locations=[World[0],World[2],World[1],World[3]])
WorldStream.filter(locations=[min_x,min_y,max_x,max_y])
