import logging
import os
from datetime import datetime

formatter = "%(levelname)s : %(asctime)s : %(message)s"
logging.basicConfig(format=formatter)

log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "log")
os.makedirs(log_dir, exist_ok=True)
filename = os.path.join(
    log_dir,
    f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
)
file_handler = logging.FileHandler(filename=filename)
file_handler.setFormatter(logging.Formatter(formatter))

LOG_NAME = "algotham"


class Logger:
    logger = logging.getLogger(LOG_NAME)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

    @staticmethod
    def d(tag: str, message: str):
        """debug log"""
        Logger.logger.debug("[%s] %s", tag, message)

    @staticmethod
    def i(tag: str, message: str):
        """infomation log"""
        Logger.logger.info("[%s] %s", tag, message)

    @staticmethod
    def e(tag: str, message: str):
        """error log"""
        Logger.logger.error("[%s] %s", tag, message)

    @staticmethod
    def w(tag: str, message: str):
        """warning log"""
        Logger.logger.warning("[%s] %s", tag, message)

    @staticmethod
    def set_level(level):
        Logger.logger.setLevel(level)
