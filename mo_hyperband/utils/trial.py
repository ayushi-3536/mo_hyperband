import numpy as np
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

    def get_status(self):
        return self._status

    def get_fitness(self):
        return self._fitness

    def finish_trial(self, fitness):
        self._fitness = fitness
        self._status = Status.COMPLETED.value
        logger.debug(f'fitness:{self._fitness}')

    def start_trial(self):
        self._status = Status.RUNNING