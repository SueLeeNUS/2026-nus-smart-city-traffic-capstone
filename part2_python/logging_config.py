"""Shared logging setup.

This helper is called ONLY from entry-point scripts (main.py, pipeline.py when
run directly, and traffic_app.py). Library modules never configure handlers;
they just use ``logging.getLogger(__name__)``.
"""

import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_file: str = "logs/pipeline.log",
    debug: bool = False,
    console_level: int = logging.INFO,
    file_mode: str = "w",
) -> None:
    """Send logs to both the console and a log file.

    debug=True turns on DEBUG messages (fine-grained values); otherwise the
    lowest level shown is INFO.
    """
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_level = logging.DEBUG if debug else logging.INFO
    if debug:
        console_level = logging.DEBUG

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    file_handler = logging.FileHandler(log_file, mode=file_mode, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    # Handlers are attached to the top-level logger of this project's modules'
    # parent (the root) once, here, so every module's logger inherits them.
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Third-party libraries (matplotlib, PIL) are very chatty at DEBUG level; keep them quiet
    # so our own DEBUG messages stay readable.
    for noisy in ("matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
