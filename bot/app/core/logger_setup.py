import json
import os
import logging
import logging.handlers


def setup_logger():
    """配置规范化的日志系统。"""
    log_level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    log_level = os.getenv("LOG_LEVEL", "info").lower()
    log_dir = os.getenv("LOG_DIR", "./logs")
    log_max_size = int(os.getenv("LOG_MAX_SIZE", 10 * 1024 * 1024))
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))

    os.makedirs(log_dir, exist_ok=True)

    log_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(module)s | %(funcName)s | %(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_map.get(log_level, logging.INFO))
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, "divination_master.log"),
        maxBytes=log_max_size,
        backupCount=log_backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(log_format)
    root_logger.addHandler(file_handler)

    return root_logger


logger = setup_logger()


def log_event(level: int, event: str, **fields) -> None:
    payload = {"event": event}
    for key, value in fields.items():
        payload[key] = value
    logger.log(level, json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
