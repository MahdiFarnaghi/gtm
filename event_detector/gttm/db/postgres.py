import json
from copy import copy
from datetime import datetime

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


class PostgresHandler:
    min_acceptable_num_words_in_tweet = 4
    expected_db_version = 3

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

        event_detection_task_table = Table('event_detection_task', meta,
                                        #    Column('id', BigInteger, Sequence(
                                        #        'seq_event_detection_task_id'), primary_key=True),
                                           Column('task_name', String(100), nullable=False, primary_key=True),
                                           Column('desc', String(
                                               500), nullable=True),
                                           Column('min_x', Numeric, nullable=False),
                                           Column('min_y', Numeric, nullable=False),
                                           Column('max_x', Numeric, nullable=False),
                                           Column('max_y', Numeric, nullable=False),
                                           Column('lang_code', String(2), nullable=False),
                                           Column('interval_min', Integer, nullable=False)
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
            if db_version == 2:
                self.create_database_schema_version03()
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


