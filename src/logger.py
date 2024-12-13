import os, uuid
import logging
from datetime import datetime

def create_logfile():
    LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d')}.log"
    LOG_FILE_DIR_PATH = "./logs"
    os.makedirs(LOG_FILE_DIR_PATH, exist_ok=True)
    LOG_FILE_PATH = os.path.join(LOG_FILE_DIR_PATH, LOG_FILE).replace("\\", "/")
    return LOG_FILE_PATH

class Logger:
    unique_id = uuid.uuid4()

    logging.basicConfig(level=logging.INFO, filename=create_logfile(),
                        format="[ %(asctime)s ] - %(unique_id)s - %(name)s - %(levelname)s - %(message)s",
                        handlers=[logging.FileHandler(create_logfile())]
                        )
    old_f = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = Logger.old_f(*args, **kwargs)
        record.unique_id = Logger.unique_id
        return record

    #logging.setLogRecordFactory(record_factory)

    def update_id(self):
        Logger.unique_id = uuid.uuid1()
        logging.setLogRecordFactory(Logger.record_factory)
        return