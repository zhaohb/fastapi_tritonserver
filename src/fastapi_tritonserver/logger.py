"""Logging configuration for TRT-LLM."""
import logging
import sys

_FORMAT = "%(levelname)s %(asctime)s.%(msecs)03d [%(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None):
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        msg = logging.Formatter.format(self, record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


_root_logger = logging.getLogger("trt-llm-server")
_default_handler = None


def get_formatter():
    return NewLineFormatter(_FORMAT, datefmt=_DATE_FORMAT)


def _setup_logger():
    print('setup trt-llm logger')
    logging.basicConfig(level = logging.INFO)
    _root_logger.setLevel(logging.INFO)
    global _default_handler
    if _default_handler is None:
        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.flush = sys.stdout.flush  # type: ignore
        _default_handler.setLevel(logging.INFO)
        _root_logger.addHandler(_default_handler)
    _default_handler.setFormatter(get_formatter())
    # Setting this will avoid the message
    # being propagated to the parent logger.
    _root_logger.propagate = False


# The logger is initialized when the module is imported.
# This is thread-safe as the module is only imported once,
# guaranteed by the Python GIL.
_setup_logger()


def init_logger(name: str):
    return logging.getLogger(name)

