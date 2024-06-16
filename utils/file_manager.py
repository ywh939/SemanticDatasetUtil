import os

def read_file_list(filelist_path):
    file_list = os.listdir(filelist_path)
    for f in file_list:
        file_path = filelist_path / f
        yield file_path