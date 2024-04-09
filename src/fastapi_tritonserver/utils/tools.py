from random import randrange
import math
import uuid

max_range = math.pow(2, 64) - 1


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def random_int64() -> int:
    return randrange(0, max_range)