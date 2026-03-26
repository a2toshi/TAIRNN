import os


def check_filename(filename):
    if os.path.exists(filename):
        raise ValueError("{} exists.".format(filename))
    return filename


def check_path(path, mkdir=False):
    """
    Checks that path is collect
    """
    if path[-1] == "/":
        path = path[:-1]

    if not os.path.exists(path):
        if mkdir:
            os.makedirs(path, exist_ok=True)
        else:
            raise ValueError("%s does not exist" % path)

    return path
