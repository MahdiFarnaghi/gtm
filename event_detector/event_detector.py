import sys
import os
from time import sleep
from gttm.ts.task_scheduler import TaskScheduler
from gttm.db.postgres_event_detection_task import PostgresHandler_EventDetectionTask
from dotenv import load_dotenv
from datetime import datetime, timedelta
from gttm.db.postgres_tweet import PostgresHandler_Tweets
from gttm.nlp import VectorizerUtil_FastText
from gttm.ioie.geodata import add_geometry
import numpy as np



class EventDetector:
    def __init__(self, check_database_threshold=60):
        """
        Initialize an EventDetector object
        """
        self.check_database_threshold = check_database_threshold

        self.db_hostname = os.getenv('DB_HOSTNAME')
        self.db_port = os.getenv('DB_PORT')
        self.db_user = os.getenv('DB_USER')
        self.db_pass = os.getenv('DB_PASS')
        self.db_database = os.getenv('DB_DATABASE')
        # self._postgres = PostgresHandler_EventDetectionTask('localhost', 5432, 'tweetsdb', 'postgres', 'postgres')
        self.postgres = PostgresHandler_EventDetectionTask(
            self.db_hostname, self.db_port,  self.db_database, self.db_user, self.db_pass)

        self.task_list = {}

        self.scheduler = TaskScheduler()
        self.scheduler.start_scheduler()

    def updates_event_detection_task(self):

        db_tasks = self.postgres.get_tasks()
        for task in db_tasks:
            sch_task = self.scheduler.get_task(task['task_name'])
            if sch_task is None:
                self.scheduler.add_task(execute_event_detection_procedure, interval_minutes=task['interval_min'], args=(
                    task['task_name'], task['min_x'], task['min_y'], task['max_x'], task['max_y'], task['look_back'], task['lang_code'],), task_id=task['task_name'])

        running_tasks_ids = self.scheduler.get_tasks_ids()
        for task_id in running_tasks_ids:
            if not any(task['task_name'] == task_id for task in db_tasks):
                self.scheduler.remove_task(task_id)

        pass

    def run(self):
        while (True):
            self.updates_event_detection_task()
            # Check for new instruction in the database
            sleep(self.check_database_threshold)


load_dotenv()
db_hostname = os.getenv('DB_HOSTNAME')
db_port = os.getenv('DB_PORT')
db_user = os.getenv('DB_USER')
db_pass = os.getenv('DB_PASS')
db_database = os.getenv('DB_DATABASE')

postgres = PostgresHandler_Tweets(db_hostname, db_port, db_database, db_user, db_pass)
vectorizer = VectorizerUtil_FastText()

def execute_event_detection_procedure(process_name, min_x, min_y, max_x, max_y, look_back_hours: int, lang_code):
    print(F"Process: {process_name}, Language: {lang_code}")
    
    global postgres, vectorizer

    end_date = datetime.now()
    start_date = end_date - timedelta(hours=int(look_back_hours))

    # Read data from database
    print("1. Read data from database.")    
    df, num = postgres.read_data_from_postgres(
        start_date=start_date,
        end_date=end_date,
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        lang=lang_code)
    
    if num <= 0:
        print('There was no record for processing.')
        return
    print(F"Number of retrieved tweets: {num}")

    # convert to geodataframe
    print("2. convert to GeoDataFrame")
    gdf = add_geometry(df) 

    # get location vectors
    print("3. Get location vectors")
    x = np.asarray(gdf.geometry.x)[:, np.newaxis]
    y = np.asarray(gdf.geometry.y)[:, np.newaxis]

    #TODO: Get time vector. Time vector should be in days format

    # Vectorzie text
    print("5. Get text vector")
    # text_vect = vectorizer.vectorize(df.c.values, lang_code)
    # print(F"Shape of the vectorized tweets: {text_vect.shape}")

    #TODO: extract clusters first textually and then based on location and time    

    

if __name__ == '__main__':
    load_dotenv()
    # event_detector = EventDetector()
    # event_detector.run()
    execute_event_detection_procedure('test', -180, -90, 180, 90, 100, 'en')

#TODO: Add prepare mathod to download everythings, including fasttext files before the main task
