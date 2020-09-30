from postgres import PostgresHandler

class event_detection_task(PostgresHandler):
    
    def __init__(self, DB_HOSTNAME, DB_PORT, DB_DATABASE, DB_USERNAME, DB_PASSWORD):
        super.__init__(DB_HOSTNAME, DB_PORT, DB_DATABASE, DB_USERNAME, DB_PASSWORD)


    def insert_event_detection_task(self, task: dict):
        self.check_db()
        
        pass

    def get_event_detection_tasks(self):
        pass
