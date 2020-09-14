from listener import MyListener
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
consumer_key = os.getenv('consumer_key')
consumer_key = os.getenv('consumer_key')
access_token = os.getenv('access_token')
access_secret = os.getenv('access_secret')
 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
api = tweepy.API(auth)

#MIN_X,MAX_X,MIN_Y,MAX_Y
World = [-180,180,-90,90]

#World
WorldStreamListener = MyListener()
WorldStreamListener.init("World")
WorldStream = Stream(auth, WorldStreamListener)
#[SWlongitude, SWLatitude, NElongitude, NELatitude]
WorldStream.filter(locations=[World[0],World[2],World[1],World[3]])
