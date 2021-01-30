import json
from copy import copy
from datetime import datetime
from sqlalchemy.sql.elements import collate
from sqlalchemy.sql.expression import null
import sqlalchemy_utils
from geoalchemy2 import Geometry
from sqlalchemy import Column
from sqlalchemy import Integer, String, BigInteger, DateTime
from sqlalchemy import MetaData
from sqlalchemy import Table, select, func
from sqlalchemy import create_engine, Numeric, Boolean, ForeignKey, Sequence
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine.url import URL
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import Session
import pandas as pd

class PostgresHandler:
    min_acceptable_num_words_in_tweet = 4
    expected_db_version = 4

    def __init__(self, DB_HOSTNAME, DB_PORT, DB_DATABASE, DB_USERNAME, DB_PASSWORD):

        self.postgres_db = {'drivername': 'postgresql',
                            'username': DB_USERNAME,
                            'password': DB_PASSWORD,
                            'host': DB_HOSTNAME,
                            'port': DB_PORT,
                            'database': DB_DATABASE}
        self.db_url = URL(**self.postgres_db)
        self.db_version = None
        self.engine = None
        self.db_is_checked = False
        self.table_tweet = None
        self.table_twitter_user = None
        self.table_hashtag = None
        self.table_tweet_hashtag = None
        self.table_event_detection_task = None
        self.table_cluster = None
        self.table_cluster_points = None

    def __del__(self):
        # if self.engine is not None:
        #     self.engine.dispose()
        pass

    def set_db_url(self, postgres_db: dict):
        self.postgres_db = postgres_db
        self.db_url = URL(**self.postgres_db)
        self.db_version = None
        self.engine = None
        self.db_is_checked = False

    def get_db_version(self):
        try:
            check_version_sql = "SELECT value FROM db_properties WHERE key='version';"
            res = self.engine.execute(check_version_sql)
            for r in res:
                return int(r[0])
        except:
            self.db_version = None
        return self.db_version

    def create_database_schema_version01(self):
        meta = MetaData(self.engine)
        property_table = Table('db_properties', meta,
                               Column('key', String, primary_key=True),
                               Column('value', String))
        task_id_seq = Sequence(
            'event_detection_task_id_seq', metadata=meta)
        event_detection_task_table = Table('event_detection_task', meta,
                                           Column('task_id', BigInteger, task_id_seq,
                                                  server_default=task_id_seq.next_value(),
                                                  primary_key=True),
                                           Column('task_name', String(
                                               100), nullable=False, unique=True),
                                           Column('desc', String(
                                               500), nullable=True),
                                           Column('min_x', Numeric,
                                                  nullable=False),
                                           Column('min_y', Numeric,
                                                  nullable=False),
                                           Column('max_x', Numeric,
                                                  nullable=False),
                                           Column('max_y', Numeric,
                                                  nullable=False),
                                           Column('look_back', Numeric,
                                                  nullable=False),
                                           Column('lang_code', String(
                                               2), nullable=False),
                                           Column('interval_min',
                                                  Integer, nullable=False),
                                           Column('active', Boolean,
                                                  default=False),
                                           )

        meta.create_all()
        property_table_sql_insert_version_1 = "INSERT INTO db_properties (key, value) " \
                                              "VALUES ('version', '{}');".format(
                                                  "1")
        self.engine.execute(property_table_sql_insert_version_1)
        pass

    def create_database_schema_version02(self):
        meta = MetaData(self.engine)

        user_table = Table('twitter_user', meta,
                           Column('id', BigInteger, primary_key=True),
                           Column('name', String(50)),
                           Column('screen_name', String(50)),
                           Column('location', String(300)),
                           Column('followers_count', Integer),
                           Column('friends_count', Integer),
                           Column('listed_count', Integer),
                           Column('favourites_count', Integer),
                           Column('statuses_count', Integer),
                           Column('geo_enabled', Boolean),
                           Column('lang', String(5)))

        hashtag_table = Table('hashtag', meta,
                              Column('id', Integer, primary_key=True,
                                     autoincrement=True),
                              Column('value', String(100), unique=True))

        tweet_table = Table('tweet', meta,
                            Column('id', BigInteger, primary_key=True),
                            Column('num', BigInteger, autoincrement=True),
                            Column('t_datetime', DateTime),
                            Column('t', BigInteger),
                            Column('created_at', DateTime),
                            Column('lang', String(5)),
                            Column('user_id', BigInteger,
                                   ForeignKey('twitter_user.id')),
                            Column('user_screen_name', String(50)),
                            Column('country', String(50), nullable=True),
                            Column('country_code', String(5), nullable=True),
                            Column('x', Numeric, nullable=True),
                            Column('y', Numeric, nullable=True),
                            Column('text', String(300)),
                            Column('c', String(300)),
                            Column('geom4326', Geometry(
                                'POINT', srid=4326), nullable=True),
                            Column('in_reply_to_status_id',
                                   BigInteger, nullable=True),
                            Column('in_reply_to_user_id',
                                   BigInteger, nullable=True),
                            Column('in_reply_to_screen_name',
                                   String(50), nullable=True),
                            Column('quoted_status_id',
                                   BigInteger, nullable=True),
                            Column('is_quote_status', Boolean, nullable=True),
                            Column('quote_count', Integer, nullable=True),
                            Column('reply_count', Integer, nullable=True),
                            Column('retweet_count', Integer, nullable=True),
                            Column('favorited', Boolean, nullable=True),
                            Column('retweeted', Boolean, nullable=True)
                            )

        hashtag_tweet_table = Table('tweet_hashtag', meta,
                                    Column('tweet_id', BigInteger, ForeignKey(
                                        'tweet.id'), primary_key=True),
                                    Column('hashtag_id', Integer, ForeignKey('hashtag.id'), primary_key=True))
        meta.create_all()
        property_table_sql_insert_version_2 = "UPDATE db_properties " \
                                              "SET value = '{}' " \
                                              "WHERE key ='version'; ".format(
                                                  "2")
        self.engine.execute(property_table_sql_insert_version_2)
        # postgis_sql = 'CREATE EXTENSION postgis;'
        # self.engine.execute(postgis_sql)
        # postgis_topology_sql = 'CREATE EXTENSION postgis_topology;'
        # self.engine.execute(postgis_topology_sql)
        pass

    def create_database_schema_version03(self):
        update_tweet_table_1 = "ALTER TABLE tweet " \
                               "ADD tag varchar(100);"
        self.engine.execute(update_tweet_table_1)

        update_tweet_table_2 = "ALTER TABLE tweet " \
                               "ADD lang_supported bool;"
        self.engine.execute(update_tweet_table_2)

        update_tweet_table_3 = "ALTER TABLE tweet " \
                               "ADD hashtags_ varchar(300);"
        self.engine.execute(update_tweet_table_3)

        property_table_sql_insert_version_3 = "UPDATE db_properties " \
                                              "SET value = '{}' " \
                                              "WHERE key ='version'; ".format(
                                                  "3")
        self.engine.execute(property_table_sql_insert_version_3)
        pass

    def create_database_schema_version04(self):
        ver = 4
        meta = MetaData(self.engine)

        cluster_id_seq = Sequence(
            'cluster_id_seq', metadata=meta)
        cluster_table = Table('cluster', meta,
                              Column('id', BigInteger, cluster_id_seq,
                                     server_default=cluster_id_seq.next_value(),
                                     primary_key=True),
                              Column('task_id', BigInteger, nullable=False),
                              Column('task_name', String(100), nullable=False),
                              Column('topic', String(2000), nullable=False),
                              Column('topic_words', String(
                                  500), nullable=False),
                              Column('lat_min', Numeric),
                              Column('lat_max', Numeric),
                              Column('lon_min', Numeric),
                              Column('lon_max', Numeric)
                              )

        cluster_point_id_seq = Sequence(
            'cluster_point_id_seq', metadata=meta)
        cluster_point = Table('cluster_point', meta,
                              Column('id', BigInteger, cluster_id_seq,
                                     server_default=cluster_point_id_seq.next_value(),
                                     primary_key=True),
                              Column('cluster_id', BigInteger, ForeignKey(
                                  'cluster.id'), nullable=False),
                              Column('longitude', Numeric, nullable=False),
                              Column('latitude', Numeric, nullable=False),
                              Column('text', String(500), nullable=False),
                              Column('date_time', DateTime, nullable=False))

        meta.create_all()
        property_table_sql_insert_version_2 = "UPDATE db_properties " \
                                              "SET value = '{}' " \
                                              "WHERE key ='version'; ".format(
                                                  str(ver))
        self.engine.execute(property_table_sql_insert_version_2)
        pass

    def load_schema(self):
        meta = MetaData(self.engine)
        self.table_tweet = Table(
            'tweet', meta, autoload=True, autoload_with=self.engine)
        self.table_twitter_user = Table('twitter_user', meta, autoload=True,
                                        autoload_with=self.engine)
        self.table_hashtag = Table('hashtag', meta, autoload=True,
                                   autoload_with=self.engine)
        self.table_tweet_hashtag = Table('tweet_hashtag', meta, autoload=True,
                                         autoload_with=self.engine)
        self.table_event_detection_task = Table('event_detection_task', meta, autoload=True,
                                                autoload_with=self.engine)

        self.table_cluster = Table('cluster', meta, autoload=True,
                                   autoload_with=self.engine)

        self.table_cluster_point = Table('cluster_point', meta, autoload=True,
                                         autoload_with=self.engine)

    def check_db(self):
        if not self.db_is_checked:
            self.engine = create_engine(
                self.db_url, isolation_level="AUTOCOMMIT", pool_size=10, max_overflow=20)

            if not sqlalchemy_utils.functions.database_exists(self.db_url):
                url = copy(make_url(self.db_url))
                url.database = 'postgres'
                engine = create_engine(url, isolation_level="AUTOCOMMIT")
                engine.execute("CREATE DATABASE {};".format(
                    self.db_url.database))
                self.engine.execute("CREATE EXTENSION postgis;")
                self.engine.execute("CREATE EXTENSION postgis_topology;")

            db_version = self.get_db_version()
            if db_version is None:
                self.create_database_schema_version01()
                db_version = self.get_db_version()
            if db_version == 1:
                self.create_database_schema_version02()
                db_version = self.get_db_version()
            if db_version == 2:
                self.create_database_schema_version03()
                db_version = self.get_db_version()
            if db_version == 3:
                self.create_database_schema_version04()
                db_version = self.get_db_version()

            # if db_version is None or db_version != self.expected_db_version:
            #     raise Exception(
            #         "Could not generate/upgrade the database to version {}".format(
            #             self.expected_db_version))
            self.load_schema()
            self.db_version = db_version
            self.db_is_checked = db_version is not None
            if self.db_version != PostgresHandler.expected_db_version:
                raise Exception("Current db version ({}) differs from expected db version ({})".format(self.db_version,
                                                                                                       PostgresHandler.expected_db_version))

        return self.db_is_checked


class PostgresHandler_Tweets(PostgresHandler):

    def __init__(self, DB_HOSTNAME, DB_PORT, DB_DATABASE, DB_USERNAME, DB_PASSWORD):
        super().__init__(DB_HOSTNAME, DB_PORT, DB_DATABASE,
                         DB_USERNAME, DB_PASSWORD)

    def read_data_from_postgres(self, start_date: datetime, end_date: datetime, min_x, min_y, max_x, max_y, table_name='tweet', tag='', verbose=False, lang=None):
        # todo: check if the table exists and catch any error
        if verbose:
            print('\tStart reading data ...')
        s_time = datetime.now()

        start = datetime(year=start_date.year, month=start_date.month,
                         day=start_date.day, hour=start_date.hour, minute=start_date.minute)
        end = datetime(year=end_date.year, month=end_date.month,
                       day=end_date.day, hour=end_date.hour, minute=end_date.minute)
        sql = F" SELECT * " \
            " FROM  {} " \
            " WHERE " \
            " t_datetime > %s AND " \
            " t_datetime <= %s AND " \
            " x >= %s AND x < %s AND" \
            " y >= %s AND y < %s ".format(table_name)

        if tag != '':
            sql = sql + " AND tag=\'{}\'".format(tag)

        if lang is not None:
            sql = sql + F" AND lang=\'{lang}\' "

        self.check_db()

        tweets = pd.read_sql_query(
            sql, self.engine, params=(start, end, min_x, max_x, min_y, max_y))
        tweets['t_datetime'] = tweets['t'].apply(pd.Timestamp.fromtimestamp)
        number_of_tweets = tweets.id.count()

        dur = datetime.now() - s_time
        if verbose:
            print('\tReading data was finished ({} seconds).'.format(dur.seconds))
        return tweets, number_of_tweets

    def extract_hashtags(self, tweet_json):
        hashtags = []
        for d in tweet_json['entities']['hashtags']:
            hashtags.append(d['text'][0:99])
        return hashtags
        pass

    def value_or_none(self, dic: dict, key: str):
        if key in dic:
            return dic[key]
        else:
            return None

    def bulk_insert_geotagged_tweets(self, tweets: list, country_code: str = '', bbox_w=0, bbox_e=0, bbox_n=0,
                                     bbox_s=0, tag='', force_insert=False):
        self.check_db()
        lst_users = []
        lst_tweets = []
        lst_tweet_ids = []
        lst_hashtags = []
        lst_tweet_hashtags = []
        for t in tweets:
            tweet_json = json.loads(t)
            x = None
            y = None
            add_it = True
            if tweet_json['coordinates'] is not None or tweet_json['geo'] is not None:
                # source: https://developer.twitter.com/en/docs/tutorials/filtering-tweets-by-location
                if tweet_json['coordinates'] is not None:
                    x = float(tweet_json['coordinates']['coordinates'][0])
                    y = float(tweet_json['coordinates']['coordinates'][1])
                else:
                    x = float(tweet_json['geo']['coordinates'][1])
                    y = float(tweet_json['geo']['coordinates'][0])
            if x is None or y is None:
                add_it = False
            if bbox_e != 0 and bbox_n != 0 and bbox_s != 0 and bbox_w:
                if not (x >= bbox_w and x <= bbox_e and y <= bbox_n and y >= bbox_s):
                    add_it = False
            elif country_code != '':
                try:
                    if country_code != tweet_json['place']['country_code']:
                        add_it = False
                except:
                    add_it = False

            cleaned_text = ''
            lang_supported = False
            num_of_words = 0
            if TextCleaner.is_lang_supported(tweet_json['lang']):
                cleaned_text, num_of_words, lang_full_name = TextCleaner.clean_text(tweet_json['text'],
                                                                                    tweet_json['lang'])
                lang_supported = True
            else:
                cleaned_text = ''
                num_of_words = len(str(tweet_json['text']).split())
                lang_supported = False

            if num_of_words < PostgresHandler.min_acceptable_num_words_in_tweet:
                add_it = False

            if add_it:
                self._add_tweet_to_insert_list(tweet_json["text"], cleaned_text, lang_supported, lst_hashtags,
                                               lst_tweet_hashtags,
                                               lst_tweet_ids, lst_tweets, lst_users, tag, tweet_json, x, y)

        if force_insert:
            if len(lst_tweet_ids) > 0:
                self.engine.execute(
                    "DELETE FROM tweet WHERE tweet.id in ({});".format(",".join(str(x) for x in lst_tweet_ids)))
        if len(lst_tweets) > 0:
            if len(lst_users) > 0:
                self.engine.execute(pg_insert(self.table_twitter_user).on_conflict_do_nothing(index_elements=['id']),
                                    lst_users)
            if len(lst_tweets) > 0:
                self.engine.execute(pg_insert(self.table_tweet).on_conflict_do_nothing(index_elements=['id']),
                                    lst_tweets)
            if len(lst_hashtags) > 0:
                self.engine.execute(pg_insert(self.table_hashtag).on_conflict_do_nothing(index_elements=['value']),
                                    lst_hashtags)
            if len(lst_tweet_hashtags) > 0:
                self.engine.execute("INSERT INTO tweet_hashtag(tweet_id, hashtag_id) "
                                    "VALUES("
                                    "   %(tweet_id)s, "
                                    "   (SELECT hashtag.id FROM hashtag WHERE hashtag.value = %(value)s)"
                                    ") ON CONFLICT (tweet_id, hashtag_id) DO NOTHING;",
                                    lst_tweet_hashtags)
        return len(lst_tweets)

    def bulk_insert_tweets(self, tweets: list, tag='', force_insert=False):
        self.check_db()
        lst_users = []
        lst_tweets = []
        lst_tweet_ids = []
        lst_hashtags = []
        lst_tweet_hashtags = []
        for t in tweets:
            tweet_json = json.loads(t)
            x = None
            y = None
            add_it = True
            if tweet_json['coordinates'] is not None or tweet_json['geo'] is not None:
                # source: https://developer.twitter.com/en/docs/tutorials/filtering-tweets-by-location
                if tweet_json['coordinates'] is not None:
                    x = float(tweet_json['coordinates']['coordinates'][0])
                    y = float(tweet_json['coordinates']['coordinates'][1])
                else:
                    x = float(tweet_json['geo']['coordinates'][1])
                    y = float(tweet_json['geo']['coordinates'][0])
            # if x is None or y is None:
            #     add_it = False

            cleaned_text = ''
            lang_supported = False
            num_of_words = 0
            _text = ''
            if 'text' in tweet_json:
                _text = tweet_json['text']
            elif 'full_text' in tweet_json:
                _text = tweet_json['full_text']
            else:
                add_it = False

            if add_it:
                if tweet_json['lang'] is not None and TextCleaner.is_lang_supported(tweet_json['lang']):
                    cleaned_text, num_of_words, lang_full_name = TextCleaner.clean_text(_text,
                                                                                        tweet_json['lang'])
                    lang_supported = True
                else:
                    cleaned_text = ''
                    num_of_words = len(str(_text).split())
                    lang_supported = False

            if num_of_words < PostgresHandler.min_acceptable_num_words_in_tweet:
                add_it = False

            if add_it:
                self._add_tweet_to_insert_list(_text, cleaned_text, lang_supported, lst_hashtags, lst_tweet_hashtags,
                                               lst_tweet_ids, lst_tweets, lst_users, tag, tweet_json, x, y)

        if force_insert:
            if len(lst_tweet_ids) > 0:
                with self.engine.begin():
                    self.engine.execute(
                        "DELETE FROM tweet WHERE tweet.id in ({});".format(",".join(str(x) for x in lst_tweet_ids)))

        if len(lst_tweets) > 0:
            with self.engine.begin():
                if len(lst_users) > 0:
                    self.engine.execute(
                        pg_insert(self.table_twitter_user).on_conflict_do_nothing(
                            index_elements=['id']),
                        lst_users)
                if len(lst_tweets) > 0:
                    self.engine.execute(pg_insert(self.table_tweet).on_conflict_do_nothing(index_elements=['id']),
                                        lst_tweets)
                if len(lst_hashtags) > 0:
                    self.engine.execute(pg_insert(self.table_hashtag).on_conflict_do_nothing(index_elements=['value']),
                                        lst_hashtags)
                if len(lst_tweet_hashtags) > 0:
                    self.engine.execute("INSERT INTO tweet_hashtag(tweet_id, hashtag_id) "
                                        "VALUES("
                                        "   %(tweet_id)s, "
                                        "   (SELECT hashtag.id FROM hashtag WHERE hashtag.value = %(value)s)"
                                        ") ON CONFLICT (tweet_id, hashtag_id) DO NOTHING;",
                                        lst_tweet_hashtags)
        return len(lst_tweets)

    def _add_tweet_to_insert_list(self, _text, cleaned_text, lang_supported, lst_hashtags, lst_tweet_hashtags,
                                  lst_tweet_ids, lst_tweets, lst_users, tag, tweet_json, x, y):
        hashtags = self.extract_hashtags(tweet_json)
        lst_tweet_ids.append(tweet_json["id"])
        lst_users.append({
            "id": tweet_json['user']['id'],
            "name": tweet_json['user']['name'],
            "screen_name": tweet_json['user']['screen_name'],
            "location": str(tweet_json['user']['location'])[0:299],
            "followers_count": tweet_json['user']['followers_count'],
            "friends_count": tweet_json['user']['friends_count'],
            "listed_count": tweet_json['user']['listed_count'],
            "favourites_count": tweet_json['user']['favourites_count'],
            "statuses_count": tweet_json['user']['statuses_count'],
            "geo_enabled": tweet_json['user']['geo_enabled'],
            "lang": tweet_json['user']['lang']})
        lst_tweets.append({
            "tag": tag,
            "lang_supported": lang_supported,
            "hashtags_": " ".join(hashtags) if len(hashtags) > 0 else '',
            "id": tweet_json["id"],
            "text": _text[0:300],
            "created_at": tweet_json['created_at'],
            "lang": tweet_json['lang'],
            "user_id": tweet_json['user']['id'],
            "user_screen_name": tweet_json['user']['screen_name'],
            "in_reply_to_status_id": tweet_json['in_reply_to_status_id'],
            'in_reply_to_user_id': tweet_json['in_reply_to_user_id'],
            "in_reply_to_screen_name": tweet_json['in_reply_to_screen_name'],
            # "quoted_status_id":tweet_json['quoted_status_id'],
            "is_quote_status": tweet_json['is_quote_status'],
            "quote_count": self.value_or_none(tweet_json, 'quote_count'),
            "reply_count": self.value_or_none(tweet_json, 'reply_count'),
            "retweet_count": self.value_or_none(tweet_json, 'retweet_count'),
            "favorited": self.value_or_none(tweet_json, 'favorited'),
            "retweeted": self.value_or_none(tweet_json, 'retweeted'),
            "country": tweet_json['place']['country'] if tweet_json['place'] is not None and
            tweet_json['place'][
                'country'] is not None else '',
            "country_code": tweet_json['place']['country_code'] if tweet_json['place'] is not None and
            tweet_json['place'][
                'country_code'] is not None else '',
            "c": cleaned_text[0:300],
            "t": datetime.strptime(tweet_json['created_at'], '%a %b %d %H:%M:%S %z %Y').timestamp(),
            "t_datetime": datetime.strptime(tweet_json['created_at'], '%a %b %d %H:%M:%S %z %Y'),
            "x": x,
            "y": y})
        [lst_hashtags.append({'value': h})
         for h in hashtags if h.strip() != ""]
        [lst_tweet_hashtags.append({'tweet_id': tweet_json["id"], 'value': h}) for h in hashtags if
         h.strip() != ""]

    def upsert_tweet(self, tweet_text: str, country_code: str = '', bbox_w=0, bbox_e=0, bbox_n=0,
                     bbox_s=0, tag='', force_insert=False) -> bool:
        self.check_db()
        tweet_json = json.loads(tweet_text)
        x = None
        y = None

        if tweet_json['coordinates'] is not None or tweet_json['geo'] is not None:
            # source: https://developer.twitter.com/en/docs/tutorials/filtering-tweets-by-location
            if tweet_json['coordinates'] is not None:
                x = float(tweet_json['coordinates']['coordinates'][0])
                y = float(tweet_json['coordinates']['coordinates'][1])
            else:
                x = float(tweet_json['geo']['coordinates'][1])
                y = float(tweet_json['geo']['coordinates'][0])
        if x is None or y is None:
            return False

        if bbox_e != 0 and bbox_n != 0 and bbox_s != 0 and bbox_w != 0:
            if not (x >= bbox_w and x <= bbox_e and y <= bbox_n and y >= bbox_s):
                return False
        elif country_code != '':
            try:
                if country_code != tweet_json['place']['country_code']:
                    return False
            except:
                return False

        # upsert: https://docs.sqlalchemy.org/en/13/dialects/postgresql.html
        cleaned_text = ''
        lang_supported = False
        num_of_words = 0
        if TextCleaner.is_lang_supported(tweet_json['lang']):
            cleaned_text, num_of_words, lang_full_name = TextCleaner.clean_text(
                tweet_json['text'], tweet_json['lang'])
            lang_supported = True
        else:
            cleaned_text = ''
            num_of_words = len(str(tweet_json['text']).split())
            lang_supported = False

        hashtags = self.extract_hashtags(tweet_json)

        if num_of_words >= PostgresHandler.min_acceptable_num_words_in_tweet:
            ins_user = pg_insert(self.table_twitter_user).values(
                id=tweet_json['user']['id'],
                name=tweet_json['user']['name'],
                screen_name=tweet_json['user']['screen_name'],
                location=str(tweet_json['user']['location'])[0:299],
                followers_count=tweet_json['user']['followers_count'],
                friends_count=tweet_json['user']['friends_count'],
                listed_count=tweet_json['user']['listed_count'],
                favourites_count=tweet_json['user']['favourites_count'],
                statuses_count=tweet_json['user']['statuses_count'],
                geo_enabled=tweet_json['user']['geo_enabled'],
                lang=tweet_json['user']['lang'],
            ).on_conflict_do_nothing(index_elements=['id'])
            res_user = self.engine.execute(ins_user)

            ins = pg_insert(self.table_tweet).values(
                tag=tag,
                lang_supported=lang_supported,
                hashtags_=" ".join(hashtags) if len(hashtags) > 0 else '',
                id=tweet_json["id"],
                text=tweet_json["text"],
                created_at=tweet_json['created_at'],
                lang=tweet_json['lang'],
                user_id=tweet_json['user']['id'],
                user_screen_name=tweet_json['user']['screen_name'],
                in_reply_to_status_id=tweet_json['in_reply_to_status_id'],
                in_reply_to_user_id=tweet_json['in_reply_to_user_id'],
                in_reply_to_screen_name=tweet_json['in_reply_to_screen_name'],
                # quoted_status_id=tweet_json['quoted_status_id'],
                is_quote_status=tweet_json['is_quote_status'],
                quote_count=self.value_or_none(tweet_json, 'quote_count'),
                reply_count=self.value_or_none(tweet_json, 'reply_count'),
                retweet_count=self.value_or_none(tweet_json, 'retweet_count'),
                favorited=self.value_or_none(tweet_json, 'favorited'),
                retweeted=self.value_or_none(tweet_json, 'retweeted'),
                country=tweet_json['place']['country'] if tweet_json['place'] is not None and tweet_json['place'][
                    'country'] is not None else '',
                country_code=tweet_json['place']['country_code'] if tweet_json['place'] is not None and
                tweet_json['place'][
                    'country_code'] is not None else '',
                c=cleaned_text,
                t=datetime.strptime(
                    tweet_json['created_at'], '%a %b %d %H:%M:%S %z %Y').timestamp(),
                t_datetime=datetime.strptime(
                    tweet_json['created_at'], '%a %b %d %H:%M:%S %z %Y'),
                x=x,
                y=y
            )
            if force_insert:
                ins = ins.on_conflict_do_update(
                    index_elements=['id'],
                    set_=dict(
                        lang_supported=lang_supported,
                        hashtags_=" ".join(hashtags) if len(
                            hashtags) > 0 else '',
                        created_at=tweet_json['created_at'],
                        lang=tweet_json['lang'],
                        user_id=tweet_json['user']['id'],
                        user_screen_name=tweet_json['user']['screen_name'],
                        text=tweet_json['text'],
                        in_reply_to_status_id=tweet_json['in_reply_to_status_id'],
                        in_reply_to_user_id=tweet_json['in_reply_to_user_id'],
                        in_reply_to_screen_name=tweet_json['in_reply_to_screen_name'],
                        # quoted_status_id=tweet_json['quoted_status_id'],
                        is_quote_status=tweet_json['is_quote_status'],
                        quote_count=self.value_or_none(
                            tweet_json, 'quote_count'),
                        reply_count=tweet_json['reply_count'],
                        retweet_count=tweet_json['retweet_count'],
                        favorited=tweet_json['favorited'],
                        retweeted=tweet_json['retweeted'],
                        country=tweet_json['place']['country'] if tweet_json['place'] is not None and
                        tweet_json['place'][
                            'country'] is not None else '',
                        country_code=tweet_json['place']['country_code'] if tweet_json['place'] is not None and
                        tweet_json['place'][
                            'country_code'] is not None else '',
                        c=cleaned_text,
                        t=datetime.strptime(
                            tweet_json['created_at'], '%a %b %d %H:%M:%S %z %Y').timestamp(),
                        t_datetime=datetime.strptime(
                            tweet_json['created_at'], '%a %b %d %H:%M:%S %z %Y'),
                        x=x,
                        y=y
                    )
                )
            else:
                ins = ins.on_conflict_do_nothing(index_elements=['id'])
            res = self.engine.execute(ins)

            session = Session()
            for h in hashtags:
                ins_hashtag = pg_insert(self.table_hashtag).values(
                    value=h
                ).on_conflict_do_nothing(index_elements=['value'])
                res_hashtag = self.engine.execute(ins_hashtag)
                hashtag_id = None
                if res_hashtag.rowcount > 0:
                    hashtag_id = res_hashtag.inserted_primary_key[0]
                else:
                    hashtag_id = session.query(
                        self.table_hashtag).filter_by(value=h).first()[0]
                # print("Hashtag id: {}".format(hashtag_id ))
                ins_tweet_hashtag = pg_insert(self.table_tweet_hashtag).values(
                    tweet_id=tweet_json["id"],
                    hashtag_id=hashtag_id
                ).on_conflict_do_nothing(index_elements=['tweet_id', 'hashtag_id'])
                res_tweet_hashtag = self.engine.execute(ins_tweet_hashtag)
                # print("Tweet_Hashtag id: {}".format(res_tweet_hashtag.inserted_primary_key[0]))
            return res.rowcount > 0
        return False

    def number_of_tweets(self):
        return self.engine.execute('SELECT count(*) FROM tweet;').scalar()

    def update_geom(self):
        updateQuery = "update public.tweet set geom4326=ST_SetSRID(ST_MakePoint(x, y), 4326);"
        self.engine.execute(updateQuery)


class PostgresHandler_EventDetection(PostgresHandler):
    def __init__(self, DB_HOSTNAME, DB_PORT, DB_DATABASE, DB_USERNAME, DB_PASSWORD):
        super().__init__(DB_HOSTNAME, DB_PORT, DB_DATABASE,
                         DB_USERNAME, DB_PASSWORD)

    def get_tasks(self) -> dict:
        tasks = []
        self.check_db()

        result = self.engine.execute(self.table_event_detection_task.select())
        for row in result:
            tasks.append(
                {
                    'task_id': row['task_id'],
                    'task_name': row['task_name'],
                    'desc': row['desc'],
                    'min_x': row['min_x'],
                    'min_y': row['min_y'],
                    'max_x': row['max_x'],
                    'max_y': row['max_y'],
                    'look_back': row['look_back'],
                    'lang_code': row['lang_code'],
                    'interval_min': row['interval_min']}
            )
        return tasks

    def delete_event_detection_tasks(self):
        self.check_db()
        self.engine.execute(self.table_event_detection_task.delete())

    def delete_event_detection_task(self, task_name):
        self.check_db()
        self.engine.execute(self.table_event_detection_task.delete().where(
            self.table_event_detection_task.c.task_name == task_name))

    def insert_event_detection_task(self, task_name, desc: str, min_x, min_y, max_x, max_y, look_back, lang_code, interval_min, force_insert=False) -> int:
        self.check_db()

        ins = pg_insert(self.table_event_detection_task).values(
            task_name=task_name[0:100],
            desc=desc[0: 500],
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
            look_back=look_back,
            lang_code=lang_code,
            interval_min=interval_min)
        if force_insert:
            ins = ins.on_conflict_do_update(
                index_elements=['task_name'],
                set_=dict(
                    task_name=task_name[0:100],
                    desc=desc[0: 500],
                    min_x=min_x,
                    min_y=min_y,
                    max_x=max_x,
                    max_y=max_y,
                    look_back=look_back,
                    lang_code=lang_code,
                    interval_min=interval_min)
            )
        else:
            ins = ins.on_conflict_do_nothing(
                index_elements=['task_id'])
            ins = ins.on_conflict_do_nothing(
                index_elements=['task_name'])
        res = self.engine.execute(ins)
        return res.lastrowid
        # return res.rowcount > 0

    def insert_clusters(self, clusters: list):

        for cluster in clusters:
            if cluster['id'] is None:
                clust = cluster.copy()
                points = clust.pop('points')
                clust.pop('id')
                ins = pg_insert(self.table_cluster).on_conflict_do_nothing(
                    index_elements=['id']).values(clust)
                res = self.engine.execute(ins)
                clust_id = res.lastrowid
                if clust_id is not None:
                    for i in range(len(points)):
                        points[i]["cluster_id"] = clust_id
                    ins_points = pg_insert(self.table_cluster_point).on_conflict_do_nothing(
                        index_elements=['id']).values(points)
                    self.engine.execute(ins_points)
                else:
                    raise Exception('cluster_id is None!')
            else:
                clust = cluster.copy()
                points = clust.pop('points')
                ins = pg_insert(self.table_cluster).values(clust)
                ins = ins.on_conflict_do_update(index_elements=['id'],
                                                set_=clust).values(clust)  # Maybe I should pop the id before
                self.engine.execute(ins)
                ins_points = pg_insert(self.table_cluster_point).on_conflict_do_nothing(
                    index_elements=['id']).values(points)
                self.engine.execute(ins_points)
        pass

    def insert_cluster(self, task_id, task_name, topic, topic_words, lat_min, lat_max, lon_min, lon_max) -> int:
        ins_user = pg_insert(self.table_cluster).values(
            task_id=task_id,
            task_name=task_name,
            topic=topic,
            topic_words=topic_words,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max
        ).on_conflict_do_nothing(index_elements=['cluster_id'])
        res_cluster = self.engine.execute(ins_user)

        return res_cluster.lastrowid
