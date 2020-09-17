import json
from copy import copy
from datetime import datetime

import sqlalchemy_utils
from geoalchemy2 import Geometry
from sqlalchemy import Column
from sqlalchemy import Integer, String, BigInteger, DateTime
from sqlalchemy import MetaData
from sqlalchemy import Table, select, func
from sqlalchemy import create_engine, Numeric, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine.url import URL
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import Session

from nlp import TextCleaner


class PostgresHandler:
    min_acceptable_num_words_in_tweet = 4
    expected_db_version = 2

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
        property_table_sql_insert_version_1 = "INSERT INTO db_properties (key, value) " \
                                              "VALUES ('version', '{}');".format(
                                                  "1")
        self.engine.execute(property_table_sql_insert_version_1)
        # postgis_sql = 'CREATE EXTENSION postgis;'
        # self.engine.execute(postgis_sql)
        # postgis_topology_sql = 'CREATE EXTENSION postgis_topology;'
        # self.engine.execute(postgis_topology_sql)
        pass

    def create_database_schema_version02(self):
        update_tweet_table_1 = "ALTER TABLE tweet " \
                               "ADD tag varchar(100);"
        self.engine.execute(update_tweet_table_1)

        update_tweet_table_2 = "ALTER TABLE tweet " \
                               "ADD lang_supported bool;"
        self.engine.execute(update_tweet_table_2)

        update_tweet_table_3 = "ALTER TABLE tweet " \
                               "ADD hashtags_ varchar(300);"
        self.engine.execute(update_tweet_table_3)

        property_table_sql_insert_version_2 = "UPDATE db_properties " \
                                              "SET value = '{}' " \
                                              "WHERE key ='version'; ".format(
                                                  "2")
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

    def check_db(self):
        if not self.db_is_checked:
            self.engine = create_engine(
                self.db_url, isolation_level="AUTOCOMMIT", pool_size=10, max_overflow=0)

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

    # def create_table(table_name):
    #     cursor = get_postgres_conn().cursor()
    #     print("Connected!")
    #     cursor.execute('DROP TABLE IF EXISTS {};'.format(
    #         table_name))
    #     get_postgres_conn().commit()
    #
    #     # [tweet['id'], t, t.timestamp(), tweet['lang'], tweet['user']['id'],
    #     #             tweet['user']['screen_name'], tweet['coordinates']['coordinates'][0],
    #     #             tweet['coordinates']['coordinates'][1], tweet['text'], str(cleaned_text)]
    #
    #     cursor.execute(
    #         'CREATE TABLE IF NOT EXISTS {}('
    #         'id serial PRIMARY KEY, tweetid bigint, t bigint, t_datetime timestamptz, '
    #         'lang varchar(5), userid bigint, userscreenname varchar(50), x numeric, y numeric, '
    #         'text varchar(300), c varchar(300), '
    #         'geom4326 geometry(POINT, 4326)'
    #         ');'.format(
    #             table_name))
    #
    #     get_postgres_conn().commit()
    #     print("Table is there.")
    #     cursor.execute('DELETE FROM {};'.format(table_name))
    #     get_postgres_conn().commit()
    #     print("Table is empty.")
    #     pass
    #
    # def insert_to_postgres(gdf: gpd.GeoDataFrame, table_name, srid, if_exist='replace'):
    #     try:
    #         # POSTGRESQL_USER = "postgres"
    #         # POSTGRESQL_PASSWORD = "post123456"
    #         # POSTGRESQL_HOST_IP = "127.0.0.1"
    #         # POSTGRESQL_PORT = "5432"
    #         # POSTGRESQL_DATABASE = "geotweets"
    #
    #         db_hostname, db_port, db_database, db_username, db_password = EnvVarSettings.get_connection_settings_from_env()
    #
    #         query_string = 'postgresql://{}:{}@{}:{}/{}'.format(db_username, db_password, db_hostname,
    #                                                             db_port, db_database)
    #         engine = create_engine(query_string, echo=False)
    #         ds = gdf.copy()
    #         ds['geom'] = ds['geometry'].apply(lambda x: WKTElement(x.wkt, srid=srid))
    #         ds.drop('geometry', 1, inplace=True)
    #         ds.to_sql(name=table_name,
    #                   con=engine,
    #                   if_exists=if_exist,
    #                   index=False, dtype={'geom': Geometry('POINT', srid=str(srid))})
    #         return True
    #
    #     except:
    #         print('-' * 60)
    #         print("Unexpected error:", sys.exc_info()[0])
    #         print('-' * 60)
    #         traceback.print_exc(file=sys.stdout)
    #         print('-' * 60)
    #         return False
