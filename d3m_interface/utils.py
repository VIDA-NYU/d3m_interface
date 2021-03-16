import shutil
import platform
from os.path import exists


def fix_path_for_docker(path):
    if platform.system() == 'Windows':
        path = path.replace('\\', '/')
    return path


def copy_folder(source_path, destination_path, remove_destination=False):
    if remove_destination and exists(destination_path):
        shutil.rmtree(destination_path)

    shutil.copytree(source_path, destination_path)
