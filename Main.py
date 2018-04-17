import fnmatch
import os

from PIL import Image


def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


for filename in find_files('./CarsID', '*'):
    try:
        img = Image.open(filename)
        img.close()
    except OSError:
        print("Removing file: %f".format(filename))
        os.remove(filename)

print("Exit...")
