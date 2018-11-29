import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))


def get_path(path='/'):
    if path is None:
        path = '/'
    if path[0] != '/':
        path = '/' + path
    return ROOT_PATH + path
