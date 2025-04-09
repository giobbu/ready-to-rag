import os

def exist_QA_files(params):
    " Check if the files exist in the output directory "
    if os.path.exists(params.out_dir_train):
        return True
    else:
        return False

