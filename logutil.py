"""
Extensions to the logging system.

Implements something like https://docs.rs/env_logger/0.9.0/env_logger/ but for
Python.
"""

import logging
import os
import sys
from typing import Union
from os import PathLike

StrPath = Union[str, PathLike[str]]

APP_NAME = 'aiopanel'
LOG_VAR = 'AIOPANEL_LOG'
DEFAULT_LOG_LEVEL = logging.INFO

LEVELS = set(logging._levelToName.keys())

def init_log_levels_from_env():
    v = os.environ.get(LOG_VAR)
    if not v:
        return
    configs = v.split(',')

    def handle_config(c: str):
        item, *rest = c.split('=')
        print(item, rest)
        if len(rest) == 1:
            # a=b
            lhs = item
            rhs = rest[0].upper()
        elif len(rest) == 0:
            # levelNameOrModuleName
            if item.upper() in LEVELS:
                lhs = ''
                rhs = item.upper()
            else:
                lhs = item
                rhs = logging.NOTSET
        else:
            raise ValueError(f'Invalid syntax in {LOG_VAR} item {item:r}')
        logger = logging.getLogger(lhs if lhs else APP_NAME)
        logger.setLevel(rhs)

    for config in configs:
        handle_config(config)


def make_logger(path: StrPath) -> logging.Logger:
    log = logging.getLogger(APP_NAME)

    # don't override settings in our logger put there before we loaded
    if not log.level:
        log.setLevel(DEFAULT_LOG_LEVEL)

    fmt = logging.Formatter(
        '{asctime} {levelname} {name}: {message}',
        datefmt='%b %d %H:%M:%S',
        style='{'
    )

    if sys.stdout.isatty():
        log.addHandler(logging.StreamHandler())

    # don't add handlers repeatedly when I use autoreload
    for handler in log.handlers:
        if isinstance(handler, logging.FileHandler) or \
                isinstance(handler, logging.StreamHandler):
            break
    else:
        hnd = logging.FileHandler(path)
        log.addHandler(hnd)

    for handler in log.handlers:
        handler.setFormatter(fmt)

    return log


def get_log() -> logging.Logger:
    return logging.getLogger(APP_NAME)


class LogMixin:
    """
    A class to expose a .log property on subclasses which logs to a separate
    stream from the rest of the program
    """

    @property
    def log(self) -> logging.Logger:
        logger = get_log().getChild(self.__class__.__name__)
        if logger.level != logging.NOTSET:
            # it was already set by something else, don't touch it
            return logger
        log_level = getattr(self, 'log_level', None)
        if log_level is not None:
            logger.setLevel(log_level)
        return logger


