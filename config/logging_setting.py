from pydantic import BaseModel, Field
from typing import List
from loguru import logger
from datetime import datetime

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

class LoggingSettings(BaseModel):
    name: str = f"{current_time}.log"
    rotation: str = "500 KB"
    retention: str = "10 days"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"

def setup_logger(path_to_save: str) -> logger:
    """
    Setup logger with specified settings.
    """
    logging = LoggingSettings()
    logger.remove()
    logger.add(
        path_to_save + "/" + logging.name,
        rotation=logging.rotation,
        retention=logging.retention,
        format=logging.format
    )
    return logger