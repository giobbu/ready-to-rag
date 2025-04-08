from loguru import logger
from config.logging_setting import setup_logger
logger = setup_logger()

def exist_QA_files(params):
    " Check if the files exist in the output directory "
    import os
    if os.path.exists(params.out_dir_train):
        logger.debug(f"File {params.out_dir_train} exists")
        return True
    else:
        logger.info(f"File {params.out_dir_train} does not exist")
        return False



