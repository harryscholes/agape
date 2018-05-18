'''Base class.
'''
from .meta import Meta

__all__ = ['Base']


class Base(object, metaclass=Meta):
    '''AGAPE base class.
    '''
    def __init__(self):
        pass
