import time
from functools import wraps


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        stop = time.time()
        sec = stop - start
        print(f"Finished '{func.__name__}' in {sec:>4.1f}s")
        return result
    return wrapper
