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

    def _check_input_types(self, data):
        if not isinstance(data, dict):
            raise TypeError("The `data` must be a dictionary of 1D Numpy arrays")

        for col_name, values in data.items():
            if not isinstance(col_name, str):
                raise TypeError('All column names must be a string')
            if not isinstance(values, np.ndarray):
                raise TypeError('All values must be a 1-D NumPy array')
            else:
                if values.ndim != 1:
                    raise ValueError('Each value must be a 1-D NumPy array')
    
    def _check_array_lengths(self, data):
        for i, values in enumerate(data.values()):
            if i == 0:
                length = len(values)
            if length != len(values):
                raise ValueError('All values must be the same length')

    def _convert_unicode_to_object(self, data):
        new_data = {}
        for col_name, values in data.items():
            if values.dtype.kind == 'U':
                new_data[col_name] = values.astype('O')
            else:
                new_data[col_name] = values
        return new_data

    def __len__(self):
        """
        Make the builtin len function work with our dataframe

        Returns
        -------
        int: the number of rows in the dataframe
        """
        return len(next(iter(self._data.values())))
                


