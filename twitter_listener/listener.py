import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import datetime
import os
import json
import sys
import traceback
import logging

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
        print("Error in parsing tweet: %s" % tweet)
        # logger.warning("Imcomplete tweet: %s", tweet)


class TwitterStreamListener(StreamListener):
    

    def on_data(self, data):
        try:
            #print("----------------------")
            self.tweetnumber+=1
            #print('Tweet number ' + str(self.tweetnumber) + ' in ' + self.area_name)
            now = datetime.datetime.now()
            filenameJson = self.output_folder + os.sep + self.area_name + os.sep + now.strftime('%Y%m%d-%H') + ".json"
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
            #traceback.print_exc(file=sys.stdout)
            #print('No GeoTag, not saved.')
            #print("----------------------")
        return True
 
    def on_error(self, status):
        print(status)
        return True

    def init(self, area_name: str, output_folder: str):
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('Saving geotagged tweets in ' + area_name+ ' started')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('\n')
        
        #self.minx = minX
        #self.maxx = maxX
        #self.miny = minY
        #self.maxy = maxY

        self.output_folder = output_folder.lower()        
        self.area_name = area_name.lower()
        self.tweetnumber = 0
        self.geotweetnumber = 0
        # self.logger = logging.getLogger(self.output_folder + os.sep + 'log')
        
