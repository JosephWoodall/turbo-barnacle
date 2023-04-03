import logging


def configure_logging(log_file_path):
    """

    :param log_file_path: 

    """
    logging.basicConfig(filename=log_file_path, level=logging.DEBUG)


def log_debug(message):
    """

    :param message: 

    """
    logging.debug(message)


def log_info(message):
    """

    :param message: 

    """
    logging.info(message)


def log_warning(message):
    """

    :param message: 

    """
    logging.warning(message)


def log_error(message):
    """

    :param message: 

    """
    logging.error(message)


def log_critical(message):
    """

    :param message: 

    """
    logging.critical(message)
