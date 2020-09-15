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
consumer_key = os.getenv('consumer_key')
consumer_secret = os.getenv('consumer_secret')
access_token = os.getenv('access_token')
access_secret = os.getenv('access_secret')
output_folder = os.getenv('TWEETS_OUPUT_FOLDER')
print('Reading the env variables was finished!')

print('Authenication ...')
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
print('Authenication was finished!')

api = tweepy.API(auth)

#MIN_X,MAX_X,MIN_Y,MAX_Y
World = [-180,180,-90,90]

print('Initializing the listener ...')
WorldStreamListener = TwitterStreamListener()
WorldStreamListener.init("World", output_folder)
WorldStream = Stream(auth, WorldStreamListener)
WorldStream.filter(locations=[World[0],World[2],World[1],World[3]])
