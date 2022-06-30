from enum import Enum
import sys
from loguru import logger
logger.configure(handlers=[{"sink": sys.stdout, "level": "DEBUG"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}


class Status(Enum):
    PENDING = 'Pending'
    RUNNING = 'Running'
    COMPLETED = 'Completed'


class Trial:
    def __init__(self, config):
        self.config = config
        self._fitness = None
        self._status = Status.PENDING.value
        self._budget = None
        self._metainfo = {}

    def get_status(self):
        return self._status

    def get_fitness(self):
        return self._fitness

    def finish_trial(self, fitness, meta_info):
        self._fitness = fitness
        self._status = Status.COMPLETED.value
        self._metainfo = meta_info
        logger.debug(f'fitness:{self._fitness}')

    def start_trial(self, budget):
        self._budget = int(budget)
        self._status = Status.RUNNING
