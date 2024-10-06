import logging
import logging.config
import os
import sys

# Create a logger
logger = logging.getLogger(__name__)

# Set the logging level
logger.setLevel(logging.DEBUG)

# Create a file handler
file_handler = logging.FileHandler('log.log')
file_handler.setLevel(logging.DEBUG)

# Create a console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add the formatter to the handlers
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Define a function to log messages
def log_message(message, level=logging.INFO):
    logger.log(level, message)

# Define a function to log errors
def log_error(message):
    logger.error(message)

# Define a function to log warnings
def log_warning(message):
    logger.warning(message)

# Define a function to log debug messages
def log_debug(message):
    logger.debug(message)

# Define a function to log info messages
def log_info(message):
    logger.info(message)

# Define a function to log critical messages
def log_critical(message):
    logger.critical(message)
