import os, logging
from chainlit.logger import logger
from datetime import datetime

def create_logfile():
    LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d')}.log"
    LOG_FILE_DIR_PATH = "./logs"
    os.makedirs(LOG_FILE_DIR_PATH, exist_ok=True)
    LOG_FILE_PATH = os.path.join(LOG_FILE_DIR_PATH, LOG_FILE).replace("\\", "/")
    return LOG_FILE_PATH

'''logging.basicConfig(level=logging.INFO, filename=create_logfile(),
                        format="[ %(asctime)s ] - %(unique_id)s - %(name)s - %(levelname)s - %(message)s",
                        handlers=[logging.FileHandler(create_logfile())]
                        )'''

formatter = logging.Formatter("[ %(asctime)s ] - %(name)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler(create_logfile())
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)