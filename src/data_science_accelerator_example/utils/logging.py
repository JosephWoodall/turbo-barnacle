import logging


def configure_logging(log_file_path):
    logging.basicConfig(filename=log_file_path, level=logging.DEBUG)


def log_debug(message):
    logging.debug(message)


def log_info(message):
    logging.info(message)


def log_warning(message):
    logging.warning(message)


def log_error(message):
    logging.error(message)


def log_critical(message):
    logging.critical(message)
