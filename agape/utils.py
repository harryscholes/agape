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


def stdout(string, object=None, file=None):
    '''Pretty(ish) print a string (and object) to STDOUT.

    # Arguments
        string: str, if `object` is provided, `string` typically describes this
        object: object, any object
        file: stream, if None, prints to STDOUT

    >>> p = "Earth"
    >>> stdout("Planet", p)
    Planet:
        Earth
    <BLANKLINE>
    >>> from io import StringIO
    >>> f = StringIO()
    >>> stdout("a", file=f)
    >>> f.getvalue()
    'a\\n\\n\\n'
    '''
    if string and object:
        print(f'{string}:\n    ', object, '\n', sep='', file=file)
    elif string:
        print(f'{string}\n\n', file=file)
