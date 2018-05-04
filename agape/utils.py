'''
Useful utilities.
'''
import os


def directory_exists(p):
    '''Checks if a directory exists.

    # Arguments
        p: str, path

    # Returns
        True: if directory path exists

    # Raises
        FileNotFoundError: if directory path does not exist
    '''
    p = os.path.expandvars(p)
    if not os.path.exists(p):
        raise FileNotFoundError("Directory does not exist", p)
    return p
