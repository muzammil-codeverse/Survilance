import logging
import sys
from datetime import datetime


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def log_stage(logger: logging.Logger, stage: str, detail: str = "") -> None:
    """Log a named pipeline stage with optional detail."""
    separator = "-" * 50
    logger.info(separator)
    logger.info(f"STAGE: {stage}")
    if detail:
        logger.info(f"       {detail}")
    logger.info(separator)


def log_dataset_stats(logger: logging.Logger, split: str, count: int, classes: list) -> None:
    """Log dataset split statistics."""
    logger.info(f"[{split.upper()}] samples={count} | classes={len(classes)}")
    for cls in classes:
        logger.debug(f"  - {cls}")
