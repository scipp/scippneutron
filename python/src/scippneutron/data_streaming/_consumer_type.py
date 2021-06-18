from enum import Enum


class ConsumerType(Enum):
    """
    Picklable way of instructing the mp.Process to
    construct real consumers or fakes for tests
    """
    REAL = 1
    FAKE = 2
