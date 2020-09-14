import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import datetime
import os
import json
import sys


 # parse data
def parse_tweet(data):

    # load JSON item into a dict
    tweet = json.loads(data)


    # check if tweet is valid
    if 'user' in tweet.keys():

        
        # classify tweet type based on metadata
        if 'retweeted_status' in tweet:
            tweet['TWEET_TYPE'] = 'retweet'

        elif len(tweet['entities']['user_mentions']) > 0:
            tweet['TWEET_TYPE'] = 'mention'

        else:
            tweet['TWEET_TYPE'] = 'tweet'

        return tweet

    else:
        logger.warning("Imcomplete tweet: %s", tweet)


class MyListener(StreamListener):
    

    def on_data(self, data):
        try:
            #print("----------------------")
            self.tweetnumber+=1
            #print('Tweet number ' + str(self.tweetnumber) + ' in ' + self.cityname)
            now = datetime.datetime.now()
            filenameJson = 'C://Twitter//' + self.cityname + '//' + now.strftime('%Y%m%d-%H') + ".json"
            directory = os.path.dirname(filenameJson)
            if not os.path.exists(directory):
                os.makedirs(directory)            
            tweet = parse_tweet(data)
            if not tweet['geo'] is None:                
                #print(tweet['text'].encode('unicode_escape'))                
                with open(filenameJson, 'a') as f:
                    f.write(data) 
                    self.geotweetnumber+=1
                    print("----------------------")
                    print('Geotagged tweet ' + str(self.geotweetnumber) + ' saved of ' + str(self.tweetnumber) + ' tweets, at ' + now.strftime('%Y%m%d-%H:%M:%S') + ".")  
                    print("----------------------")
                    return True                   
                
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        #print('No GeoTag, not saved.')
        #print("----------------------")
        return True
 
    def on_error(self, status):
        print(status)
        return True

    def init(self, cityName):
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('Saving geotagged tweets in ' + cityName+ ' started')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('\n')
        #self.minx = minX
        #self.maxx = maxX
        #self.miny = minY
        #self.maxy = maxY
        self.cityname = cityName
        self.tweetnumber = 0
        self.geotweetnumber = 0
        
