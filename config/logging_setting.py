from pydantic import BaseModel, Field
from typing import List
from loguru import logger

class LoggingSettings(BaseModel):
    name: str = "app.log"
    rotation: str = "500 KB"
    retention: str = "10 days"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"

def setup_logger():
    logging = LoggingSettings()
    logger.remove()
    logger.add(
        logging.name,
        rotation=logging.rotation,
        retention=logging.retention,
        format=logging.format
    )
    return logger