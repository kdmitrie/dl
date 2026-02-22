import signal
from typing import Callable, Optional

class TimeoutError(Exception):
    pass

def handle_timeout(signum, frame):
    raise TimeoutError


def run_with_timeout(task: Callable, timeout: int=5, on_timeout: Optional[Callable] = None):
    signal.signal(signal.SIGALRM, handle_timeout)
    signal.alarm(timeout)

    try:
        task()
    except TimeoutError:
        print("Task took too long to finish")
        if on_timeout is not None and callable(on_timeout):
            on_timeout()
    finally:
        signal.alarm(0)
