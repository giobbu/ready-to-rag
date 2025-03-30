from loguru import logger 

def exist_QA_files(config):
    " Check if the files exist in the output directory "
    import os
    if os.path.exists(config.out_dir_train):
        logger.debug(f"File {config.out_dir_train} exists")
        return True
    else:
        logger.info(f"File {config.out_dir_train} does not exist")
        return False



