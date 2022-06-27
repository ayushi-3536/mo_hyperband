import numpy as np
from enum import Enum

class Status(Enum):
    PENDING = 'Pending'
    RUNNING = 'Running'
    COMPLETED = 'Completed'

class Trial:
    def __init__(self, config):
        self.config = config
        self._fitness = np.inf
        self._status = Status.PENDING

    def get_status(self):
        return self._status

    def get_fitness(self):
        return self._fitness

    def finish_trial(self, fitness):
        self._fitness = fitness
        self._status = Status.COMPLETED

    def start_trial(self):
        self._status = Status.RUNNING