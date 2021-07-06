import logging


class LoggingLevels:

    @staticmethod
    def logging_level(name: str):
        name = name.upper()
        return {
            'CRITICAL': 50,
            'ERROR': 40,
            'WARNING': 30,
            'INFO': 20,
            'DEBUG': 10,
            'NOTSET': 0
        }.get(name, 20)

    @staticmethod
    def print_and_log_info(logger, text: str):
        print(text)
        logger.info(text)
