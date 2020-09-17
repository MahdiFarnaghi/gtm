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
from db.postgres import PostgresHandler

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
            self.tweetnumber += 1
            tweet = parse_tweet(data)
            now = datetime.datetime.now()

            if self.save_geotweets and tweet['geo'] is None:
                return

            if self.save_data_mode == 'DB':
                self.geotweetnumber += 1
                if self.postgres.upsert_tweet(data):
                    self.write_tweet_saved(
                        self.geotweetnumber, self.tweetnumber, self.save_data_mode, now)
                else:
                    print('The tweet could not be saved.')
            else:
                #print('Tweet number ' + str(self.tweetnumber) + ' in ' + self.area_name)
                filenameJson = self.output_folder + os.sep + self.area_name.lower() + \
                    os.sep + now.strftime('%Y%m%d-%H') + ".json"
                directory = os.path.dirname(filenameJson)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # print(tweet['text'].encode('unicode_escape'))
                with open(filenameJson, 'a') as f:
                    f.write(data)
                    self.geotweetnumber += 1
                    self.write_tweet_saved(
                        self.geotweetnumber, self.tweetnumber, self.save_data_mode, now)
                    return True

        except BaseException as e:
            print("Error on_data: %s" % str(e))
            # traceback.print_exc(file=sys.stdout)
            #print('No GeoTag, not saved.')
            # print("----------------------")
        return True

    def write_tweet_saved(self, geotweetnumber, tweetnumber, save_data_mode, dt):
        print("----------------------")
        print(F"{save_data_mode}: {str(geotweetnumber)} tweet saved out of {str(tweetnumber)}, at " +
               dt.strftime("%Y%m%d-%H:%M:%S"))
        print(self.postgres.number_of_tweets())
        print("----------------------")

    def on_error(self, status):
        # print(status)
        return True

    def init(self,  area_name: str, output_folder: str, postgres: PostgresHandler, save_data_mode='FILE', save_geotweets=True):
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('Saving geotagged tweets in ' + area_name + ' started')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('\n')

        #self.minx = minX
        #self.maxx = maxX
        #self.miny = minY
        #self.maxy = maxY

        self.save_data_mode = save_data_mode
        self.output_folder = output_folder
        self.area_name = area_name
        self.tweetnumber = 0
        self.geotweetnumber = 0
        self.postgres = postgres
        self.save_geotweets = save_geotweets

        if self.save_data_mode == 'DB':
            if self.postgres is None:
                raise Exception('postgres parameter is None.')
        elif self.save_data_mode == 'FILE':
            if self.output_folder is None or self.output_folder == '' or self.area_name is None or self.area_name == '':
                raise Exception(
                    'output_folder parameter, area_name parameter, or both are None/empty.')
        else:
            raise Exception(
                F'The specified save_data_mode ({self.save_data_mode}) is not supported.')
        # self.logger = logging.getLogger(self.output_folder + os.sep + 'log')
