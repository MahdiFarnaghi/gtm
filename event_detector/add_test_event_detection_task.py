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
# postgres.delete_event_detection_tasks()
# postgres.insert_event_detection_task('task 1 NYC', 'desc ...', -76, 39, 71.5, 42, 36, 'en', 5, False)

print('Inserted')

# postgres.delete_event_detection_tasks()
postgres.insert_event_detection_task('task 2 London', 'desc ...', -1, 51, 1, 52, 36, 'en', 5, False)

# postgres.insert_event_detection_task('task 2 Lisbon', 'desc ...', -9.5, 38.5, -9, 39, 36, 'pt', 5, False)

#AREA_NAME=london
#MIN_X=-1
#MAX_X=1
#MIN_Y=51
#MAX_Y=52

# postgres.insert_event_detection_task('task 1 in sweden', 'desc ...', 1, 2, 3, 4, 'sv', 5, False)
# print('Inserted')
# sleep(delay)
# postgres.insert_event_detection_task('task 2 in eng', 'desc ...', 1, 2, 3, 4, 'en', 5, False)
# print('Inserted')
# sleep(delay)
# postgres.delete_event_detection_task('task 1 in sweden')
# print('Deleted')
# sleep(delay)
# postgres.insert_event_detection_task('task 3 in sweden', 'desc ...', 1, 2, 3, 4, 'sv', 5, False)
# print('Inserted')
# tasks = postgres.get_tasks()
# [print(F"Taks name: {t['task_name']}, Min X: {t['min_x']}") for t in tasks]

# scheduler = TaskScheduler()
# scheduler.start_scheduler()
# def print_a_sentence(sentence):
#     print(sentence)
# scheduler.add_task(print_a_sentence, interval_minutes=1, args=('Task 1',), task_id='task1')
# scheduler.add_task(print_a_sentence, interval_minutes=3, args=('Task 2',), task_id='task2')
# sleep(20)
# scheduler.remove_task('task2')

# while (True):
#     check_for_updates()
#     # Check for new instruction in the database
#     sleep(60)
