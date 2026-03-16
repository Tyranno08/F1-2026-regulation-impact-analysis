# src/logger.py

import logging
import os
from datetime import datetime


def get_logger(name: str, log_to_file: bool = True) -> logging.Logger:
    """
    Creates and returns a configured logger.

    Args:
        name:        The name of the logger, typically the module name.
        log_to_file: Whether to also write logs to a file in logs/ directory.

    Returns:
        A configured logging.Logger instance.
    """

    # Create logs directory if it does not exist
    if log_to_file:
        logs_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs"
        )
        os.makedirs(logs_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers if logger is called multiple times
    if logger.handlers:
        return logger

    # Define log format
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler — shows INFO and above in terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler — writes DEBUG and above to a log file
    if log_to_file:
        log_filename = os.path.join(
            logs_dir,
            f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler = logging.FileHandler(log_filename, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger