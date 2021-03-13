import sys
import os
from time import sleep
from gttm.ts.task_scheduler import TaskScheduler
from gttm.db import PostgresHandler_EventDetection
from dotenv import load_dotenv


load_dotenv()

postgres = PostgresHandler_EventDetection(
    os.getenv('DB_HOSTNAME'), 
    os.getenv('DB_PORT'),
    os.getenv('DB_DATABASE'),
    os.getenv('DB_USER'),
    os.getenv('DB_PASS'))

postgres.check_db()

delay = 10
postgres.delete_event_detection_tasks()
postgres.insert_event_detection_task('task 1 NYC', 'desc ...', -76, 39, 71.5, 42, 36, 'en', 2, True)
# postgres.insert_event_detection_task('task 2 London', 'desc ...', -1, 51, 1, 52, 36, 'en', 5, True)
# postgres.insert_event_detection_task('task 2 Lisbon', 'desc ...', -9.5, 38.5, -9, 39, 36, 'pt', 5, True)

print('Inserted')