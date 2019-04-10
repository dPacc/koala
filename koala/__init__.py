import numpy as np 

__version__ = '0.0.1'


class DataFrame:

    def __init__(self, data):
        """
        A DataFrame holds two dimensional heterogeneous data. It can
        be created by passing a dictionary of Numpy arrays associated with 
        each string.

        Parameters
        ----------
        data: dict
            A dictionary of string mapped to Numpy arrays. The key will become the
            column name.

        """
        # Check for the correct input types
        self._check_input_types(data)

        # Check for equal array lengths
        self._check_array_lengths(data)

        # Convert unicode arrays to object
        self._data = self._convert_unicode_to_data(data)

        # Allow for special methods for strings
        self.str = StringMethods(self)
        self._add_docs()

        
