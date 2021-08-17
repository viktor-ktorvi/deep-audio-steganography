import os
import glob


def delete_all_files_in_folder(PATH):
    files = glob.glob(os.path.join(PATH, '*'))
    for f in files:
        os.remove(f)
