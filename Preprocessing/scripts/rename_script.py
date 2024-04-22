import os

def _(directory):
    files = os.listdir(directory)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)))
    for i, file in enumerate(files, start=1):
        extension = os.path.splitext(file)[1]
        new_name = f"{i}{extension}"
        os.rename(os.path.join(directory, file), os.path.join(directory, new_name))

_("D:\\Semester 8\\AudioGeneration\\dataset")
