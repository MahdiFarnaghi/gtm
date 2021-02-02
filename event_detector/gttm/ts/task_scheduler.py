from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.executors.pool import ProcessPoolExecutor

from pytz import utc


class TaskScheduler:
    def __init__(self):
        jobs_database_name = 'jobs.sqlite'
        jobstores = {
           # 'default': SQLAlchemyJobStore(url=F'sqlite:///{jobs_database_name}')
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
            jobstores=jobstores,
            executors=executors, 
            job_defaults=job_defaults, 
            timezone=utc)              

    def get_task(self, task_id):
        return self.scheduler.get_job(task_id)
    
    def get_tasks_ids(self):
        task_ids = []
        jobs = self.scheduler.get_jobs()
        for j in jobs:
            task_ids.append(j.id)
        
        return task_ids

    def start_scheduler(self):
        self.scheduler.start()

    def stop_scheduler(self):
        self.scheduler.shutdown()

    def add_task(self, task_func, interval_minutes, args, task_id):
        print('Adding an interval task')
        self.scheduler.add_job(
            task_func,
            IntervalTrigger(minutes=interval_minutes),
            args=args,
            id=str(task_id))
        print('Adding the interval task finished')

    def remove_task(self, id):
        print(F'Removing task (Id: {id})')
        self.scheduler.remove_job(id)
        print('The task removed.')
