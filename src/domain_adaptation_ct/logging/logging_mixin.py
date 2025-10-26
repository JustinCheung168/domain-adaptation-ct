import datetime
import logging
import os

def init_logging(log_dir: str):
    """Set up global logging format with timestamps. Assume a logger already exists."""

    # Create a timestamped filename
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"log_{current_time}.log")
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    # Attach to the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    print(f"Logging to {log_filename}")

class LogMixin:
    """Make this one of the parents of a class to have logs created in the class appear."""
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
