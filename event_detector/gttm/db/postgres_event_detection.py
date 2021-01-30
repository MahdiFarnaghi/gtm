# from gttm.db import PostgresHandler_Tweets
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


# class PostgresHandler_EventDetection(PostgresHandler_Tweets):
#     def __init__(self, DB_HOSTNAME, DB_PORT, DB_DATABASE, DB_USERNAME, DB_PASSWORD):
#         super().__init__(DB_HOSTNAME, DB_PORT, DB_DATABASE,
#                          DB_USERNAME, DB_PASSWORD)

#     def get_tasks(self) -> dict:
#         tasks = []
#         self.check_db()

#         result = self.engine.execute(self.table_event_detection_task.select())
#         for row in result:
#             tasks.append(
#                 {
#                     'task_id': row['task_id'],
#                     'task_name': row['task_name'],
#                     'desc': row['desc'],
#                     'min_x': row['min_x'],
#                     'min_y': row['min_y'],
#                     'max_x': row['max_x'],
#                     'max_y': row['max_y'],
#                     'look_back': row['look_back'],
#                     'lang_code': row['lang_code'],
#                     'interval_min': row['interval_min']}
#             )
#         return tasks

#     def delete_event_detection_tasks(self):
#         self.check_db()
#         self.engine.execute(self.table_event_detection_task.delete())

#     def delete_event_detection_task(self, task_name):
#         self.check_db()
#         self.engine.execute(self.table_event_detection_task.delete().where(
#             self.table_event_detection_task.c.task_name == task_name))

#     def insert_event_detection_task(self, task_name, desc: str, min_x, min_y, max_x, max_y, look_back, lang_code, interval_min, force_insert=False) -> int:
#         self.check_db()

#         ins = pg_insert(self.table_event_detection_task).values(
#             task_name=task_name[0:100],
#             desc=desc[0: 500],
#             min_x=min_x,
#             min_y=min_y,
#             max_x=max_x,
#             max_y=max_y,
#             look_back=look_back,
#             lang_code=lang_code,
#             interval_min=interval_min)
#         if force_insert:
#             ins = ins.on_conflict_do_update(
#                 index_elements=['task_name'],
#                 set_=dict(
#                     task_name=task_name[0:100],
#                     desc=desc[0: 500],
#                     min_x=min_x,
#                     min_y=min_y,
#                     max_x=max_x,
#                     max_y=max_y,
#                     look_back=look_back,
#                     lang_code=lang_code,
#                     interval_min=interval_min)
#             )
#         else:
#             ins = ins.on_conflict_do_nothing(
#                 index_elements=['task_id'])
#             ins = ins.on_conflict_do_nothing(
#                 index_elements=['task_name'])
#         res = self.engine.execute(ins)
#         return res.lastrowid
#         # return res.rowcount > 0

#     def insert_clusters(self, clusters: list):
        
#         for cluster in clusters:
#             if cluster['id'] is None:
#                 clust = cluster.copy()                
#                 points = clust.pop('points')
#                 clust.pop('id')                
#                 ins = pg_insert(self.table_cluster).on_conflict_do_nothing(index_elements=['id']).values(clust)
#                 res = self.engine.execute(ins)
#                 clust_id = res.lastrowid;
#                 if clust_id is not None:
#                     for i in range(len(points)):
#                         points[i]["cluster_id"] = clust_id
#                     ins_points = pg_insert(self.table_cluster_point).on_conflict_do_nothing(index_elements=['id']).values(points)
#                     self.engine.execute(ins_points)
#                 else:
#                     raise Exception('cluster_id is None!')
#             else:
#                 clust = cluster.copy()                
#                 points = clust.pop('points')
#                 ins = pg_insert(self.table_cluster).values(clust)
#                 ins = ins.on_conflict_do_update(index_elements=['id'],
#                                                 set_=clust).values(clust) # Maybe I should pop the id before
#                 self.engine.execute(ins)
#                 ins_points = pg_insert(self.table_cluster_point).on_conflict_do_nothing(index_elements=['id']).values(points)
#                 self.engine.execute(ins_points)        
#         pass
    
#     def insert_cluster(self, task_id, task_name, topic, topic_words, lat_min, lat_max, lon_min, lon_max) -> int:
#         ins_user = pg_insert(self.table_cluster).values(
#             task_id=task_id,
#             task_name=task_name,
#             topic=topic,
#             topic_words=topic_words,
#             lat_min=lat_min,
#             lat_max=lat_max,
#             lon_min=lon_min,
#             lon_max=lon_max
#         ).on_conflict_do_nothing(index_elements=['cluster_id'])
#         res_cluster = self.engine.execute(ins_user)
        
#         return res_cluster.lastrowid
    
    
        
