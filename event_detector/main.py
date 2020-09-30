import sys
from time import sleep
from task_scheduler import TaskScheduler


print('='*60)
print('Starting the EVENT DETECTION program ...')
print('='*60)
print('\tPython version: {sys.version}')
print("\tVersion info: {sys.version_info}")
print('='*60)
print('\tSetting up the Scheduler ...')

def check_for_updates():
    print('\tDatabase is checked.')
    pass

if __name__ == '__main__':
    scheduler = TaskScheduler()
    scheduler.start_scheduler()
    def print_a_sentence(sentence):
        print(sentence)
    scheduler.add_task(print_a_sentence, interval_minutes=1, args=('Task 1',), task_id='task1')
    scheduler.add_task(print_a_sentence, interval_minutes=3, args=('Task 2',), task_id='task2')
    sleep(20)
    scheduler.remove_task('task2')
    
    while (True):
        check_for_updates()
        # Check for new instruction in the database        
        sleep(60)