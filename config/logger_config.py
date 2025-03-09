import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import os
import time

os.makedirs("logs", exist_ok=True)

# Clear existing handlers (in case of module reload)
logger = logging.getLogger()
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

logger.setLevel(logging.DEBUG)

# Console handler - show WARNING by default (can be changed with --verbose or --quiet)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_formatter = logging.Formatter("%(message)s")  # Simplified for console
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Detailed debug log with rotation based on size (10 MB)
file_handler = RotatingFileHandler(
    "logs/detailed_logs.log",
    mode="a",
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=10,
)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s - %(message)s"
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Processing log with daily rotation
proc_handler = TimedRotatingFileHandler(
    "logs/processing.log",
    when="midnight",
    backupCount=30,  # Keep a month of daily logs
    encoding="utf-8",
)
proc_handler.setLevel(logging.INFO)
proc_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
proc_handler.setFormatter(proc_formatter)
proc_handler.suffix = "%Y-%m-%d"  # Use date as suffix for rotated files
logger.addHandler(proc_handler)

# Error log for easy access to just errors
error_handler = RotatingFileHandler(
    "logs/errors.log",
    mode="a",
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=5,
)
error_handler.setLevel(logging.ERROR)
error_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)
error_handler.setFormatter(error_formatter)
logger.addHandler(error_handler)

logger.info("Logger initialized at %s", time.strftime("%Y-%m-%d %H:%M:%S"))
