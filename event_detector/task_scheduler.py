from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.executors.pool import ProcessPoolExecutor

from pytz import utc


class TaskScheduler:
    def __init__(self):
        jobstores = {
            'default': SQLAlchemyJobStore(url='sqlite:///jobs.sqlite')
        }
        executors = {
            'default': {'type': 'threadpool', 'max_workers': 20},
            'processpool': ProcessPoolExecutor(max_workers=5)
        }
        job_defaults = {
            'coalesce': False,
            'max_instances': 3
        }
        self.scheduler = BackgroundScheduler()
        self.scheduler.configure(
            executors=executors, job_defaults=job_defaults, timezone=utc)

    def start_scheduler(self):
        self.scheduler.start()

    def stop_scheduler(self):
        self.scheduler.shutdown()

    def add_task(self, task_func, interval_minutes, args, task_id):
        print('\tAdding an interval task')
        self.scheduler.add_job(
            task_func,
            IntervalTrigger(seconds=interval_minutes),
            args=args,
            id=task_id)
        print('\tAdding the interval task finished')

    def remove_task(self, id):
        print(F'\tRemoving task (Id: {id})')
        self.scheduler.remove_job(id)
        print('\tThe task removed.')
