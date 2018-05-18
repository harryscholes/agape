'''Metaclasses.
'''


class Meta(type):
    '''AGAPE metaclass.
    '''
    def __new__(cls, clsname, superclasses, attributedict):
        return type.__new__(cls, clsname, superclasses, attributedict)
