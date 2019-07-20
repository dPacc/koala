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

    @property
    def columns(self):
        """
        _data holds column names mapped to arrays
        take advantage of internal ordering of dictionaries to
        put columns in correct order in list. Only works in 3.6+

        Returns
        -------
        list of column names
        """
        return list(self._data)

    @columns.setter
    def columns(self, columns):
        """
        Must supply a list of columns as strings the same length
        as the current DataFrame

        Parameters
        ----------
        columns: list of strings

        Returns
        -------
        Nones
        """
        if not isinstance(columns, list):
            raise TypeError('New columns must be a list')
        if len(columns) != len(self.columns):
            raise ValueError(f'New column length must be {len(self._data)}')
        else:
            for col in columns:
                if not isinstance(col, str):
                    raise TypeError('New column names must be strings')
        if len(columns) != len(set(columns)):
            raise ValueError('Column names must be unique')

        new_data = dict(zip(columns, self._data.values()))
        self._data = new_data


    @property
    def shape(self):
        """
        Returns
        -------
        two-item tuple of number of rows and columns
        """
        return len(self), len(self.columns)


    def _repr_html_(self):
        html += '<table><thead><tr><th></th>'
        for col in self.columns:
            html += f"<th>{col:10}</th>"

        html += '</tr></thead>'
        html += "<tbody>"

        only_head = False
        num_head = 10
        num_tail = 10
        if len(self) <= 20:
            only_head = True
            num_head = len(self)

        for i in range(num_head):
            html += f'<tr><td><strong>{i}</strong></td>'
            for col, values in self._data.items():
                kind = values.dtype.kind
                if kind == 'f':
                    html += f'<td>{values[i]:10.3f}</td>'
                elif kind == 'b':
                    html += f'<td>{values[i]}</td>'
                elif kind == 'O':
                    v = values[i]
                    if v is None:
                        v = 'None'
                    html += f'<td>{v:10}</td>'
                else:
                    html += f'<td>{values[i]:10}</td>'
            html += '</tr>'

        if not only_head:
            html += '<tr><strong><td>...</td></strong>'
            for i in range(len(self.columns)):
                html += '<td>...</td>'
            html += '</tr>'
            for i in range(-num_tail, 0):
                html += f'<tr><td><strong>{len(self) + i}</strong></td>'
                for col, values in self._data.items():
                    kind = values.dtype.kind
                    if kind == 'f':
                        html += f'<td>{values[i]:10.3f}</td>'
                    elif kind == 'b':
                        html += f'<td>{values[i]}</td>'
                    elif kind  == 'O':
                        v = values[i]
                        if v is None:
                            v = 'None'
                        html += f'<td>{v:10}</td>'
                    else:
                        html += f'<td>{values[i]:10}</td>'
                html += '</tr>'

        html += '</tbody></table>'
        return html


    @property
    def values(self):
        """
        Return a single 2D numpy array of the underlying data
        """
        return np.column_stack(self._data.values())


    @property
    def dtypes(self):
        """
        Returns a two column dataframe of column names
        in one and their data type in the other
        """
        DTYPE_NAME = {'O': 'string', 'i': 'int', 'f': 'float', 'b': 'bool'}
        col_arr = np.array(self.columns)
        dtypes = []
        for values in self._data.values():
            kind = values.dtype.kind
            dtype = DTYPE_NAME[kind]
            dtypes.appedn(dtype)

        return DataFrame({'Column Name': col_arr, 'Data Type': np.array(dtypes)})


    @property
    def __getitem__(self, item):
        """
        Returns a subset of the original DataFrame
        """

        if isinstance(item, str):
            return DataFrame({item: self._data[item]})

        if isinstance(item, list):
            return DataFrame({col: self._data[col] for col in item})

        if isinstance(item, DataFrame):
            if item.shape[1] != 1:
                raise ValueError('Can only pass one column DataFrame for the selection')

           bool_arr = next    (iter(item._data.values()))
           if bool_arr.dtype.kind != 'b':
               raise TypeError('DataFrame must be a boolean')

            new_data = {}
            for col, values in self._data.items():
                new_data[col] = values[bool_arr]
            return DataFrame(new_data)

        if isinstance(item, tuple):
            return self._getitem_tuple(item)
        else:
            raise TypeError('Select with either a string, a list or a row and column'
                            'simultaneous selection')

    def _getitem_tuple(self, item):
        # Simultaneous selction of rows and cols
        if len(item) != 2:
            raise valueError('Pass either a single string or a two-tem tuple inside the selection operator')

        row_selection = item[0]
        col_selection = item[1]
        if isinstance(row_selection, int):
            row_selection = [row_selection]
        elif isinstance(row_selection, DataFrame):
            if row_selection.shape[1] != 1:
                raise ValueError('Can only pass a one column DataFrame for selection')
            row_selection = next(iter(row_selection._data.values()))
            if row_selection.dtype.kind != 'b':
                raise TypeError('DataFrame must be a boolean')
        elif not isninstance(row_selection, (list, slice)):
            raise TypeError('Row selection must be either an int, slice, list or a DataFrame')

        if isinstance(col_selection, int):
            col_selection = [self.columns[col_selection]]
        elif isinstance(col_selection, str):
            col_selection = [col_selection]
        elif isinstance(col_selection, list):
            new_col_selection = []
            for col in col_selection:
                if isinstance(col, int):
                    new_col_selection.append(self.columns[col])
                else:
                    new_col_selection.append(col)
            col_selection = new_col_selection
        elif isinstance(col_selection, slice):
            start = col_selection.start
            stop = col_selection.stop
            step = col_selection.step
            if isinstance(start, str):
                start = self.columns.index(col_selection.start)
            if isinstance(stop, str):
                stop = self.columns.index(col_selection.stop) + 1

            col_selection = self.columns[start:stop:step]

        else:
            raise TypeError('Column selection must be either an int, string, list or slice')

        new_data = {}
        for col in col_selection:
            new_data[col] = self._data[col][row_selection]
        return DataFrame(new_data)


    def _ipython_key_completions_(self):
        # Allows for tab completion when dealing with DataFrames
        return self.columns

    def __setitem__(self, key, value):
        # Adds a new column or overwrites an old column
        if not isinstance(key, str):
            raise NotImplementedError('Only able to set a single column')

        if isinstance(value, np.ndarray):
            if value.ndim != 1:
                raise ValueError('Setting array must be 1D')
            if len(value) != len(self):
                raise ValueError('Setting DataFrame must be one column')
        elif isinstance(value, DataFrame):
            if value.shape[1] != 1:
                raise ValueError('Setting DataFrame must be one column')
            if len(value) != len(self):
                raise ValueError('Setting and Callinf DataFrames must be the same length')
            value = next(iter(value._data.values()))
        elif isinstance(value, (int, str, float, bool)):
            value = np.repeat(value, len(self))
        else:
            raise TypeError('Setting value muse either be a numpy array, '
                            'DataFrame, integer, string, float or boolean')

        if value.dtype.kind == 'U':
            value = value.astype('O')

        self._data[key] = value

    def head(self, n=5):
        """
        Returns the first 5 rows
        """
        return self[:n, :]

    def tail(self, n=5):
        """
        Returns the last 5 rows
        """
        return self[-n:, :]

    #### Aggregation Methods ###
    def min(self):
        return self._agg(np.min)

    def max(self):
        return self._agg(np.max)

    def mean(self):
        return self._agg(np.mean)

    def median(self):
        return self._agg(np.median)

    def sum(self):
        return self._agg(np.sum)

    def var(self):
        return self._agg(np.var)

    def std(self):
        return self._agg(np.std)

    def all(self):
        return self._agg(np.any)

    def argmax(self):
        return self._agg(np.argmax)

    def argmin(self):
        return self._agg(np.argmin)

    def _agg(self, aggfunc):
        """
        Generic aggregation function that is applied to each column

        Returns a dataframe
        """
        new_data = {}
        for col, values in  self._data.items():
            try:
                val = aggfunc(values)
            except TypeError:
                continue
            new_data[col] = np.array([val])
        return DataFrame(new_data)

    def isna(self):
        """
        Determines whether each value in the DataFrame is missing or not

        Returns a DataFrame of booleans, the same size as the calling DataFrame
        """
        new_data = {}
        for col, values in self._data.items():
            kind = values.dtype.kind
            if kind == 'O':
                new_data[col] = values == None
            else:
                new_data[col] = np.isnan(values)
        return DataFrame(new_data)


    def count(self):
        """
        Counts the number of non-missing values per column

        Returns a DataFrame
        """
        new_data = {}
        df =self.isna()
        length = len(self)
        for col, values in df._data.items():
            val = lenght - values.sum()
            new_data[col] = np.array([val])
        return DataFrame(new_data)

    def unique(self):
        """
        Finds the unique values of each column

        Returns a list of one column DataFrames
        """
        dfs = []
        for col, values in self._data.items():
            uniques = np.unique(values)
            dfs.append(DataFrame({col: uniques}))
        if len(dfs) == 1:
            return dfs[0]
        return dfs

    def nunique(self):
        """
        Find the number of unique values in each column

        Returns a DataFrame
        """
        new_data = {}
        for col, value in self._data.items():
            new_data[col] = np.array([len(np.unique(value))])
        return DataFrame(new_data)

    def value_counts(self, normalize=False):
        """
        Returns the frequency of each unique value for each column

        Returns a list of DataFrames or a single DataFrame if one column

        """
        dfs = []
        for col, values in self._data.items():
            keys, raw_counts = np.unique(values, return_counts=True)

            order = np.argsort(-raw_counts)
            keys = keys[order]
            raw_counts = raw_counts[order]

            if normalize:
                raw_counts = raw_counts  raw_counts.sum()
            df = DataFrame({col: keys, 'count': raw_counts})
            dfs.append(df)
        if len(dfs) == 1:
            return dfs[0]
        return dfs

    def rename(self, columns):
        """
        Renames columns in the DataFrame

        Returns a DataFrame
        """
        if not isinstance(columns, dict):
            raise TypeError('`columns` must be a dictionary')

        new_data = {}
        for col, values in self._data.items():
            new_data[columns.get(col, col)] = values
        return DataFrame(new_data)

    def drop(self, columns):
        """
        Drops one or more columsn from a DataFrame

        Returns a DataFrame

        """
        if isinstance(columns, str):
            columns = [columns]
        elif not isinstance(columns, list):
            raise TypeError('`columns` must be either a string or a list')
        new_data = {}
        for col, values in self._data.items():
            if col not in columns:
                new_data[col] = values
        return DataFrame(new_data)


    ### Non-Aggregation Methods

    def abs(self):
        """
        Take the absolute values of each value in the DataFrame
        Reutrns a DataFrame
        """
        return self._non_agg(np.abs)

    def cummin(self):
        """
        Finds the cumulative minimum by column

        Returns a DataFrame
        """
        return self._non_agg(np.minimum.accumulate)

    def cummax(self):
        """
        Finds the cumulative maximum by column

        Returns a DataFrame
        """
        return self._non_agg(np.maximum.accumulate)

    def cumsum(self):
        """
        Finds the cumulative sum by column

        Returns a DataFrame
        """
        return self._non_agg(np.cumsum)

    def clip(self, lower=None, upper=None):
        """
        All values less than lower will be set to lower
        All values freater than upper will be set to upper

        Returns a DataFrame
        """
        return seld._non_agg(np.clip, a_min=lower, a_max=upper)

    def round(self, n):
        """
        Rounds values to the nearest n decimals

        Returns a DataFrame
        """
        return self._non_agg(np.round, 'if', decimals=n)

    def copy(self):
        """
        Copies the DataFrame

        Returns a DataFrame
        """
        return self._non_agg(np.copy)

    def _non_agg(self, funcname, kinds='bif', **kwargs):
        """
        It is a generic non-aggregation function

        Returns a DataFrame
        """
        new_data = {}
        for col, values in self._data.items():
            if values.dtype.kind in kinds:
                values = funcname(values, **kwargs)
            else:
                values = values.copy()
            new_data[col] = values
        return DataFrame(new_data)

    def diff(self, n=1):
        """
        Takes the difference between the current value and the nth value above it

        Returns a DataFrame
        """
        def func(values):
            values = values.astype('float')
            values_shifted = np.roll(values, n)
            values = values - values_shifted
            if n >= 0:
                values[:n] = np.NAN
            else:
                values[:n] = np.NAN
            return values
        return self._non_agg(func)

    def pct_change(self, n=1):
        """
        Takes the percentage difference between the current value and the nth value above it

        Returns a DataFrame
        """
        def func(values):
            values = values.astype('float')
            values_shifted = np.roll(values, n)
            values = values - values_shifted
            if n >= 0:
                values[:n] = np.NAN
            else:
                values[n:] = np.NAN
            return values / values_shifted
        return self._non_agg(func)


    ### Arithmatic and Comparison Operators

    def __add__(self, other):
        return self._oper('__add__', other)

    def __radd__(self, other):
        return self._oper('__radd__', other)

    def __sub__(self, other):
        return self._oper('__sub__', other)

    def __rsub__(self, other):
        return self._oper('__rsub__', other)

    def __mul__(self, other):
        return self._oper('__mul__', other)

    def __rmul__(self, other):
        return self._oper('__rmul__', other)

    def __truediv__(self, other):
        return self._oper('__truediv__', other)

    def __rtruediv__(self, other):
        return self._oper('__rtruediv__', other)

    def __floordiv__(self, other):
        return self._oper('__floordiv__', other)

    def __rfloordiv__(self, other):
        return self._oper('__rfloordiv__', other)

    def __pow__(self, other):
        return self._oper('__pow__', other)

    def __rpow__(self, other):
        return self._oper('__rpow__', other)

    def __gt__(self, other):
        return self.__oper('__gt__', other)

    def __lt__(self, other):
        return self.__oper('__lt__', other)

    def __ge__(self, other):
        return self.__oper('__ge__', other)

    def __le__(self, other):
        return self.__oper('__le__', other)

    def __ne__(self, other):
        return self.__oper('__ne__', other)

    def __eq__(self, other):
        return self.__oper('__eq__', other)

    def _oper(self, op, other):
        """
        Generic operator method

        Returns a DataFrame
        """
        if isinstance(other, DataFrame):
            if other.shape[1] != 1:
                raise ValueError('`other` must be a one-column DataFrame')
            other = next(iter(other._data.values()))
        new_data = {}
        for col, values in self._data.items():
            func = getattr(values, op)
            new_data[col] = func(other)
        return DataFrame(new_data)

    def sort_values(self, by, asc=True):
        """
        Sort the DataFrame by one or more values

        Returns a DataFrame
        """
        if isinstance(by, str):
            order = np.argsort(self._data[by])
        elif isinstance(by, list):
            cols = [self._data[col for col in by[::-1]]]
            order = np.lexsort(cols)
        else:
            raise TypeError('`by` must be a str or a list')

        if not asc:
            order = order[::-1]
        return self[order.tolist(), :]

    def sample(self, n=None, frac=None, replace=False, seed=None):
        """
        Randomly samples rows to the DataFrame

        Returns a DataFrame
        """
        if seed:
            np.random.seed(seed)
        if frac is not None:
            if frac <= 0:
                raise ValueError('`frac` must be positive')
            n = int(frac * len(self))
        if n is not None:
            if not isinstance(n, int):
                raise TypeError('`n` must be an int')
            rows = np.random.choice(np.arange(len(self)), size=n, replace=replace).tolist()
        return self[rows, :]

    def pivot_table(self, rows=None, columns=None, values=None, aggfunc=None):
        """
        Creates a pivot table from one or two grouping columns

        Returns a DataFrame
        """
        if rows is None and columns is None:
            raise ValueError('`rows` or `columns` cannot both be `none`')

        if values is not None:
            val_data = self._data[values]
            if afffunc is None:
                raise ValueError('You must provide `aggfunc` when `values` is provided')
        else:
            if aggfunc is None:
                aggfun = 'size'
                val_data = np.empty(len(self))
            else:
                raise ValueError('You cannot provide `aggfunc` when values is None')

        if rows is not None:
            row_data = self._data[rows]
        if columsn is not None:
            col_data = self._data[columns]

        if rows is None:
            pivot_type = 'columns'
        elif columns is None:
            pivot_type = 'rows'
        else:
            pivot_type = 'all'

        from collections import defaultdict
        d = defaultdict(list)
        if pivot_type == 'columns':
            for group, val in zip(col_data, val_data):
                d[gorup].append(val)
        elif pivot_type == 'rows':
            for group, val in zip(row_data, val_data):
                d[group].append(val)
        else:
            for group1, group2, val in zip(row_data, col_data, val_data):
                d[(group1, group2)].append(val)

        agg_dict = {}
        for group, vals in d.items():
            arr = np.array(vals)
            func = getattr(np, aggfunc)
            agg_dict[group] = func(arr)

        new_data = {}
        if pivot_type == 'columns':
            for col_name in sorted(agg_dict):
                value = agg_dict[col_name]
                new_data[col_name] = np.array([value])
        elif pivot_type == 'rows':
            row_array = np.array(list(agg_dict.keys()))
            val_array = np.array(list(agg_dict.values()))

            order = np.argsort(row_array)
            new_data[rows] = row_array[order]
            new_data[aggfunc] = val_array[order]
        else:
            row_set = set()
            col_set = set()
            for group in agg_dict:
                row_set.add(group[0])
                col_set.add(group[1])
            row_list = sorted(row_set)
            col_list = sorted(col_set)
            new_data = {}
            new_data[rows] = np.array(row_list)
            for col in col_list:
                new_vals = []
                for row in row_list:
                    new_val = agg_dict.get((row, col), np.nan)
                    new_vals.append(new_val)
                new_data[col] = np.array(new_vals)
            return DataFrame(new_data)

        def _add_docs(self):
            agg_names = ['min', 'max', 'mean', 'median', 'sum', 'var'
                        'std', 'any', 'all', 'argmax', 'argmin']
            agg_doc = \
            """
            Find the {} of each column

            Returns a DataFrame
            """
            for name in agg_names:
                getattr(DataFrame, name).__doc__ = agg+doc.format(name)

    class StringMethods:

        def __init__(self, df):
            self._df = df

        def capitalize(self, col):
            return self._str_method(str.capitalize, col)

        def center(self, col, width, fillchar=None):
            if fillchar is None:
                fillchar = ' '
            return self._str_method(str.center, col, width, fillchar)

        def count(self, col, sub, start=None, stop=None):
            return self._str_method(str.count, col, sub, start, stop)

        def endswith(self, col, suffix, start=None, stop=None):
            return self._str_method(str.endswith, col, suffix, start, stop)

        def startswith(self, col, suffix, start=None, stop=None):
            return self._str_method(str.startswith, col, suffix, start, stop)

        def find(self, col, sub, start=None, stop=None):
            return self._str_method(str.find, col, sub, start, stop)

        def len(self, col):
            return self._str_method(str.__len__, col)

        def get(self, col, item):
            return self._str_method(Str.__getitem__, col, item)

        def index(self, col, sub, start=None, stop=None):
            return self,_str_method(str.index, col, sub, start, stop)

        def isalnum(self, col):
            return self_str_method(str.isalnum, col)

        def isalpha(self, col):
            return self._str_method(str.isalpha, col)

        def isdecimal(self, col):
            return self._str_method(str.isdecimal, col)

        def islower(self, col):
            return self._str_method(str.islower, col)

        def isnumeric(self, col):
            return self._str_method(str.isnumeric, col)

        def isspace(self, col):
            return self._str_method(str.isspace, col)

        def istitle(self, col):
            return self._str_method(str.istitle, col)

        def isupper(self, col):
            return self._str_method(str.isupper, col)

        def lstrip(self, col, chars):
            return self._str_method(str.lstrip, col)

        def rstrip(self, col, chars):
            return self._str_method(str.rstrip, col)

        def strip(self, col, chars):
            return self._str_method(str.strip, col, chars)
