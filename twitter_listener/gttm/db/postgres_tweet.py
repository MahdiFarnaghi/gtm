# from gttm.db.postgres import PostgresHandler
# import json
# from copy import copy
# from datetime import datetime

# import sqlalchemy_utils
# from geoalchemy2 import Geometry
# from sqlalchemy import Column
# from sqlalchemy import Integer, String, BigInteger, DateTime
# from sqlalchemy import MetaData
# from sqlalchemy import Table, select, func
# from sqlalchemy import create_engine, Numeric, Boolean, ForeignKey, Sequence
# from sqlalchemy.dialects.postgresql import insert as pg_insert
# from sqlalchemy.engine.url import URL
# from sqlalchemy.engine.url import make_url
# from sqlalchemy.orm import Session
# from gttm.nlp import TextCleaner

# class PostgresHandler_Tweets(PostgresHandler):

#     def __init__(self, DB_HOSTNAME, DB_PORT, DB_DATABASE, DB_USERNAME, DB_PASSWORD):
#         super().__init__(DB_HOSTNAME, DB_PORT, DB_DATABASE,
#                          DB_USERNAME, DB_PASSWORD)

#     def read_data_from_postgres(self, start_date: datetime, end_date: datetime, min_x, min_y, max_x, max_y, table_name='tweet', tag='', verbose=False):
#         # todo: check if the table exists and catch any error
#         if verbose:
#             print('\tStart reading data ...')
#         s_time = datetime.now()

#         start = datetime(year=start_date.year, month=start_date.month,
#                          day=start_date.day, hour=start_date.hour)
#         end = datetime(year=end_date.year, month=end_date.month,
#                        day=end_date.day, hour=end_date.hour)
#         sql = F" SELECT * " \
#             " FROM  {table_name} " \
#             " WHERE " \
#             " t_datetime > %s AND " \
#             " t_datetime <= %s AND " \
#             " x >= %s AND x < %s " \
#             " y >= %s AND y < %s "

#         if tag != '':
#             sql = sql + " AND tag=\'{}\'".format(tag)

#         self.check_db()

#         tweets = pd.read_sql_query(
#             sql, self.engine, params=(start, end, min_x, max_x, min_y, max_y))
#         tweets['t_datetime'] = tweets['t'].apply(pd.Timestamp.fromtimestamp)
#         number_of_tweets = tweets.id.count()

#         dur = datetime.now() - s_time
#         if verbose:
#             print('\tReading data was finished ({} seconds).'.format(dur.seconds))
#         return tweets, number_of_tweets

    
#     def extract_hashtags(self, tweet_json):
#         hashtags = []
#         for d in tweet_json['entities']['hashtags']:
#             hashtags.append(d['text'][0:99])
#         return hashtags
#         pass

#     def value_or_none(self, dic: dict, key: str):
#         if key in dic:
#             return dic[key]
#         else:
#             return None

#     def bulk_insert_geotagged_tweets(self, tweets: list, country_code: str = '', bbox_w=0, bbox_e=0, bbox_n=0,
#                                      bbox_s=0, tag='', force_insert=False):
#         self.check_db()
#         lst_users = []
#         lst_tweets = []
#         lst_tweet_ids = []
#         lst_hashtags = []
#         lst_tweet_hashtags = []
#         for t in tweets:
#             tweet_json = json.loads(t)
#             x = None
#             y = None
#             add_it = True
#             if tweet_json['coordinates'] is not None or tweet_json['geo'] is not None:
#                 # source: https://developer.twitter.com/en/docs/tutorials/filtering-tweets-by-location
#                 if tweet_json['coordinates'] is not None:
#                     x = float(tweet_json['coordinates']['coordinates'][0])
#                     y = float(tweet_json['coordinates']['coordinates'][1])
#                 else:
#                     x = float(tweet_json['geo']['coordinates'][1])
#                     y = float(tweet_json['geo']['coordinates'][0])
#             if x is None or y is None:
#                 add_it = False
#             if bbox_e != 0 and bbox_n != 0 and bbox_s != 0 and bbox_w:
#                 if not (x >= bbox_w and x <= bbox_e and y <= bbox_n and y >= bbox_s):
#                     add_it = False
#             elif country_code != '':
#                 try:
#                     if country_code != tweet_json['place']['country_code']:
#                         add_it = False
#                 except:
#                     add_it = False

#             cleaned_text = ''
#             lang_supported = False
#             num_of_words = 0
#             if TextCleaner.is_lang_supported(tweet_json['lang']):
#                 cleaned_text, num_of_words, lang_full_name = TextCleaner.clean_text(tweet_json['text'],
#                                                                                     tweet_json['lang'])
#                 lang_supported = True
#             else:
#                 cleaned_text = ''
#                 num_of_words = len(str(tweet_json['text']).split())
#                 lang_supported = False

#             if num_of_words < PostgresHandler.min_acceptable_num_words_in_tweet:
#                 add_it = False

#             if add_it:
#                 self._add_tweet_to_insert_list(tweet_json["text"], cleaned_text, lang_supported, lst_hashtags,
#                                                lst_tweet_hashtags,
#                                                lst_tweet_ids, lst_tweets, lst_users, tag, tweet_json, x, y)

#         if force_insert:
#             if len(lst_tweet_ids) > 0:
#                 self.engine.execute(
#                     "DELETE FROM tweet WHERE tweet.id in ({});".format(",".join(str(x) for x in lst_tweet_ids)))
#         if len(lst_tweets) > 0:
#             if len(lst_users) > 0:
#                 self.engine.execute(pg_insert(self.table_twitter_user).on_conflict_do_nothing(index_elements=['id']),
#                                     lst_users)
#             if len(lst_tweets) > 0:
#                 self.engine.execute(pg_insert(self.table_tweet).on_conflict_do_nothing(index_elements=['id']),
#                                     lst_tweets)
#             if len(lst_hashtags) > 0:
#                 self.engine.execute(pg_insert(self.table_hashtag).on_conflict_do_nothing(index_elements=['value']),
#                                     lst_hashtags)
#             if len(lst_tweet_hashtags) > 0:
#                 self.engine.execute("INSERT INTO tweet_hashtag(tweet_id, hashtag_id) "
#                                     "VALUES("
#                                     "   %(tweet_id)s, "
#                                     "   (SELECT hashtag.id FROM hashtag WHERE hashtag.value = %(value)s)"
#                                     ") ON CONFLICT (tweet_id, hashtag_id) DO NOTHING;",
#                                     lst_tweet_hashtags)
#         return len(lst_tweets)

#     def bulk_insert_tweets(self, tweets: list, tag='', force_insert=False):
#         self.check_db()
#         lst_users = []
#         lst_tweets = []
#         lst_tweet_ids = []
#         lst_hashtags = []
#         lst_tweet_hashtags = []
#         for t in tweets:
#             tweet_json = json.loads(t)
#             x = None
#             y = None
#             add_it = True
#             if tweet_json['coordinates'] is not None or tweet_json['geo'] is not None:
#                 # source: https://developer.twitter.com/en/docs/tutorials/filtering-tweets-by-location
#                 if tweet_json['coordinates'] is not None:
#                     x = float(tweet_json['coordinates']['coordinates'][0])
#                     y = float(tweet_json['coordinates']['coordinates'][1])
#                 else:
#                     x = float(tweet_json['geo']['coordinates'][1])
#                     y = float(tweet_json['geo']['coordinates'][0])
#             # if x is None or y is None:
#             #     add_it = False

#             cleaned_text = ''
#             lang_supported = False
#             num_of_words = 0
#             _text = ''
#             if 'text' in tweet_json:
#                 _text = tweet_json['text']
#             elif 'full_text' in tweet_json:
#                 _text = tweet_json['full_text']
#             else:
#                 add_it = False

#             if add_it:
#                 if tweet_json['lang'] is not None and TextCleaner.is_lang_supported(tweet_json['lang']):
#                     cleaned_text, num_of_words, lang_full_name = TextCleaner.clean_text(_text,
#                                                                                         tweet_json['lang'])
#                     lang_supported = True
#                 else:
#                     cleaned_text = ''
#                     num_of_words = len(str(_text).split())
#                     lang_supported = False

#             if num_of_words < PostgresHandler.min_acceptable_num_words_in_tweet:
#                 add_it = False

#             if add_it:
#                 self._add_tweet_to_insert_list(_text, cleaned_text, lang_supported, lst_hashtags, lst_tweet_hashtags,
#                                                lst_tweet_ids, lst_tweets, lst_users, tag, tweet_json, x, y)

#         if force_insert:
#             if len(lst_tweet_ids) > 0:
#                 with self.engine.begin():
#                     self.engine.execute(
#                         "DELETE FROM tweet WHERE tweet.id in ({});".format(",".join(str(x) for x in lst_tweet_ids)))

#         if len(lst_tweets) > 0:
#             with self.engine.begin():
#                 if len(lst_users) > 0:
#                     self.engine.execute(
#                         pg_insert(self.table_twitter_user).on_conflict_do_nothing(
#                             index_elements=['id']),
#                         lst_users)
#                 if len(lst_tweets) > 0:
#                     self.engine.execute(pg_insert(self.table_tweet).on_conflict_do_nothing(index_elements=['id']),
#                                         lst_tweets)
#                 if len(lst_hashtags) > 0:
#                     self.engine.execute(pg_insert(self.table_hashtag).on_conflict_do_nothing(index_elements=['value']),
#                                         lst_hashtags)
#                 if len(lst_tweet_hashtags) > 0:
#                     self.engine.execute("INSERT INTO tweet_hashtag(tweet_id, hashtag_id) "
#                                         "VALUES("
#                                         "   %(tweet_id)s, "
#                                         "   (SELECT hashtag.id FROM hashtag WHERE hashtag.value = %(value)s)"
#                                         ") ON CONFLICT (tweet_id, hashtag_id) DO NOTHING;",
#                                         lst_tweet_hashtags)
#         return len(lst_tweets)

#     def _add_tweet_to_insert_list(self, _text, cleaned_text, lang_supported, lst_hashtags, lst_tweet_hashtags,
#                                   lst_tweet_ids, lst_tweets, lst_users, tag, tweet_json, x, y):
#         hashtags = self.extract_hashtags(tweet_json)
#         lst_tweet_ids.append(tweet_json["id"])
#         lst_users.append({
#             "id": tweet_json['user']['id'],
#             "name": tweet_json['user']['name'],
#             "screen_name": tweet_json['user']['screen_name'],
#             "location": str(tweet_json['user']['location'])[0:299],
#             "followers_count": tweet_json['user']['followers_count'],
#             "friends_count": tweet_json['user']['friends_count'],
#             "listed_count": tweet_json['user']['listed_count'],
#             "favourites_count": tweet_json['user']['favourites_count'],
#             "statuses_count": tweet_json['user']['statuses_count'],
#             "geo_enabled": tweet_json['user']['geo_enabled'],
#             "lang": tweet_json['user']['lang']})
#         lst_tweets.append({
#             "tag": tag,
#             "lang_supported": lang_supported,
#             "hashtags_": " ".join(hashtags) if len(hashtags) > 0 else '',
#             "id": tweet_json["id"],
#             "text": _text[0:300],
#             "created_at": tweet_json['created_at'],
#             "lang": tweet_json['lang'],
#             "user_id": tweet_json['user']['id'],
#             "user_screen_name": tweet_json['user']['screen_name'],
#             "in_reply_to_status_id": tweet_json['in_reply_to_status_id'],
#             'in_reply_to_user_id': tweet_json['in_reply_to_user_id'],
#             "in_reply_to_screen_name": tweet_json['in_reply_to_screen_name'],
#             # "quoted_status_id":tweet_json['quoted_status_id'],
#             "is_quote_status": tweet_json['is_quote_status'],
#             "quote_count": self.value_or_none(tweet_json, 'quote_count'),
#             "reply_count": self.value_or_none(tweet_json, 'reply_count'),
#             "retweet_count": self.value_or_none(tweet_json, 'retweet_count'),
#             "favorited": self.value_or_none(tweet_json, 'favorited'),
#             "retweeted": self.value_or_none(tweet_json, 'retweeted'),
#             "country": tweet_json['place']['country'] if tweet_json['place'] is not None and
#             tweet_json['place'][
#                 'country'] is not None else '',
#             "country_code": tweet_json['place']['country_code'] if tweet_json['place'] is not None and
#             tweet_json['place'][
#                 'country_code'] is not None else '',
#             "c": cleaned_text[0:300],
#             "t": datetime.strptime(tweet_json['created_at'], '%a %b %d %H:%M:%S %z %Y').timestamp(),
#             "t_datetime": datetime.strptime(tweet_json['created_at'], '%a %b %d %H:%M:%S %z %Y'),
#             "x": x,
#             "y": y})
#         [lst_hashtags.append({'value': h})
#          for h in hashtags if h.strip() != ""]
#         [lst_tweet_hashtags.append({'tweet_id': tweet_json["id"], 'value': h}) for h in hashtags if
#          h.strip() != ""]

#     def upsert_tweet(self, tweet_text: str, country_code: str = '', bbox_w=0, bbox_e=0, bbox_n=0,
#                      bbox_s=0, tag='', force_insert=False) -> bool:
#         self.check_db()
#         tweet_json = json.loads(tweet_text)
#         x = None
#         y = None

#         if tweet_json['coordinates'] is not None or tweet_json['geo'] is not None:
#             # source: https://developer.twitter.com/en/docs/tutorials/filtering-tweets-by-location
#             if tweet_json['coordinates'] is not None:
#                 x = float(tweet_json['coordinates']['coordinates'][0])
#                 y = float(tweet_json['coordinates']['coordinates'][1])
#             else:
#                 x = float(tweet_json['geo']['coordinates'][1])
#                 y = float(tweet_json['geo']['coordinates'][0])
#         if x is None or y is None:
#             return False

#         if bbox_e != 0 and bbox_n != 0 and bbox_s != 0 and bbox_w != 0:
#             if not (x >= bbox_w and x <= bbox_e and y <= bbox_n and y >= bbox_s):
#                 return False
#         elif country_code != '':
#             try:
#                 if country_code != tweet_json['place']['country_code']:
#                     return False
#             except:
#                 return False

#         # upsert: https://docs.sqlalchemy.org/en/13/dialects/postgresql.html
#         cleaned_text = ''
#         lang_supported = False
#         num_of_words = 0
#         if TextCleaner.is_lang_supported(tweet_json['lang']):
#             cleaned_text, num_of_words, lang_full_name = TextCleaner.clean_text(
#                 tweet_json['text'], tweet_json['lang'])
#             lang_supported = True
#         else:
#             cleaned_text = ''
#             num_of_words = len(str(tweet_json['text']).split())
#             lang_supported = False

#         hashtags = self.extract_hashtags(tweet_json)

#         if num_of_words >= PostgresHandler.min_acceptable_num_words_in_tweet:
#             ins_user = pg_insert(self.table_twitter_user).values(
#                 id=tweet_json['user']['id'],
#                 name=tweet_json['user']['name'],
#                 screen_name=tweet_json['user']['screen_name'],
#                 location=str(tweet_json['user']['location'])[0:299],
#                 followers_count=tweet_json['user']['followers_count'],
#                 friends_count=tweet_json['user']['friends_count'],
#                 listed_count=tweet_json['user']['listed_count'],
#                 favourites_count=tweet_json['user']['favourites_count'],
#                 statuses_count=tweet_json['user']['statuses_count'],
#                 geo_enabled=tweet_json['user']['geo_enabled'],
#                 lang=tweet_json['user']['lang'],
#             ).on_conflict_do_nothing(index_elements=['id'])
#             res_user = self.engine.execute(ins_user)

#             ins = pg_insert(self.table_tweet).values(
#                 tag=tag,
#                 lang_supported=lang_supported,
#                 hashtags_=" ".join(hashtags) if len(hashtags) > 0 else '',
#                 id=tweet_json["id"],
#                 text=tweet_json["text"],
#                 created_at=tweet_json['created_at'],
#                 lang=tweet_json['lang'],
#                 user_id=tweet_json['user']['id'],
#                 user_screen_name=tweet_json['user']['screen_name'],
#                 in_reply_to_status_id=tweet_json['in_reply_to_status_id'],
#                 in_reply_to_user_id=tweet_json['in_reply_to_user_id'],
#                 in_reply_to_screen_name=tweet_json['in_reply_to_screen_name'],
#                 # quoted_status_id=tweet_json['quoted_status_id'],
#                 is_quote_status=tweet_json['is_quote_status'],
#                 quote_count=self.value_or_none(tweet_json, 'quote_count'),
#                 reply_count=self.value_or_none(tweet_json, 'reply_count'),
#                 retweet_count=self.value_or_none(tweet_json, 'retweet_count'),
#                 favorited=self.value_or_none(tweet_json, 'favorited'),
#                 retweeted=self.value_or_none(tweet_json, 'retweeted'),
#                 country=tweet_json['place']['country'] if tweet_json['place'] is not None and tweet_json['place'][
#                     'country'] is not None else '',
#                 country_code=tweet_json['place']['country_code'] if tweet_json['place'] is not None and
#                 tweet_json['place'][
#                     'country_code'] is not None else '',
#                 c=cleaned_text,
#                 t=datetime.strptime(
#                     tweet_json['created_at'], '%a %b %d %H:%M:%S %z %Y').timestamp(),
#                 t_datetime=datetime.strptime(
#                     tweet_json['created_at'], '%a %b %d %H:%M:%S %z %Y'),
#                 x=x,
#                 y=y
#             )
#             if force_insert:
#                 ins = ins.on_conflict_do_update(
#                     index_elements=['id'],
#                     set_=dict(
#                         lang_supported=lang_supported,
#                         hashtags_=" ".join(hashtags) if len(
#                             hashtags) > 0 else '',
#                         created_at=tweet_json['created_at'],
#                         lang=tweet_json['lang'],
#                         user_id=tweet_json['user']['id'],
#                         user_screen_name=tweet_json['user']['screen_name'],
#                         text=tweet_json['text'],
#                         in_reply_to_status_id=tweet_json['in_reply_to_status_id'],
#                         in_reply_to_user_id=tweet_json['in_reply_to_user_id'],
#                         in_reply_to_screen_name=tweet_json['in_reply_to_screen_name'],
#                         # quoted_status_id=tweet_json['quoted_status_id'],
#                         is_quote_status=tweet_json['is_quote_status'],
#                         quote_count=self.value_or_none(
#                             tweet_json, 'quote_count'),
#                         reply_count=tweet_json['reply_count'],
#                         retweet_count=tweet_json['retweet_count'],
#                         favorited=tweet_json['favorited'],
#                         retweeted=tweet_json['retweeted'],
#                         country=tweet_json['place']['country'] if tweet_json['place'] is not None and
#                         tweet_json['place'][
#                             'country'] is not None else '',
#                         country_code=tweet_json['place']['country_code'] if tweet_json['place'] is not None and
#                         tweet_json['place'][
#                             'country_code'] is not None else '',
#                         c=cleaned_text,
#                         t=datetime.strptime(
#                             tweet_json['created_at'], '%a %b %d %H:%M:%S %z %Y').timestamp(),
#                         t_datetime=datetime.strptime(
#                             tweet_json['created_at'], '%a %b %d %H:%M:%S %z %Y'),
#                         x=x,
#                         y=y
#                     )
#                 )
#             else:
#                 ins = ins.on_conflict_do_nothing(index_elements=['id'])
#             res = self.engine.execute(ins)

#             session = Session()
#             for h in hashtags:
#                 ins_hashtag = pg_insert(self.table_hashtag).values(
#                     value=h
#                 ).on_conflict_do_nothing(index_elements=['value'])
#                 res_hashtag = self.engine.execute(ins_hashtag)
#                 hashtag_id = None
#                 if res_hashtag.rowcount > 0:
#                     hashtag_id = res_hashtag.inserted_primary_key[0]
#                 else:
#                     hashtag_id = session.query(
#                         self.table_hashtag).filter_by(value=h).first()[0]
#                 # print("Hashtag id: {}".format(hashtag_id ))
#                 ins_tweet_hashtag = pg_insert(self.table_tweet_hashtag).values(
#                     tweet_id=tweet_json["id"],
#                     hashtag_id=hashtag_id
#                 ).on_conflict_do_nothing(index_elements=['tweet_id', 'hashtag_id'])
#                 res_tweet_hashtag = self.engine.execute(ins_tweet_hashtag)
#                 # print("Tweet_Hashtag id: {}".format(res_tweet_hashtag.inserted_primary_key[0]))
#             return res.rowcount > 0
#         return False

#     def number_of_tweets(self):
#         return self.engine.execute('SELECT count(*) FROM tweet;').scalar()

#     def update_geom(self):
#         updateQuery = "update public.tweet set geom4326=ST_SetSRID(ST_MakePoint(x, y), 4326);"
#         self.engine.execute(updateQuery)
