import numpy as np
from numpy.testing import assert_array_equal
import pytest

import koala as koa
from tests import assert_df_equals

pytestmark = pytest.mark.filterwarnings("ignore")

a = np.array(['a', 'b', 'c'])
b = np.array(['c', 'd', None])
c = np.random.rand(3)
d = np.array([True, False, True])
e = np.array([1, 2, 3])
df = koa.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e})


class TestSelection:

    def test_one_column(self):
        assert_array_equal(df['a'].values[:, 0], a)
        assert_array_equal(df['c'].values[:, 0], c)

    def test_multiple_columns(self):
        cols = ['a', 'c']
        df_result = df[cols]
        df_answer = pdc.DataFrame({'a': a, 'c': c})
        assert_df_equals(df_result, df_answer)

    def test_simple_boolean(self):
        bool_arr = np.array([True, False, False])
        df_bool = pdc.DataFrame({'col': bool_arr})
        df_result = df[df_bool]
        df_answer = pdc.DataFrame({'a': a[bool_arr], 'b': b[bool_arr],
                                   'c': c[bool_arr], 'd': d[bool_arr],
                                   'e': e[bool_arr]})
        assert_df_equals(df_result, df_answer)

        with pytest.raises(ValueError):
            df_bool = pdc.DataFrame({'col': np.array[1, 2, 3]})

    def test_one_column_tuple(self):
        assert_df_equals(df[:, 'a'], pdc.DataFrame({'a': a}))

    def test_multiple_columns_tuple(self):
        cols = ['a', 'c']
        df_result = df[:, cols]
        df_answer = pdc.DataFrame({'a': a, 'c': c})
        assert_df_equals(df_result, df_answer)

    def test_int_selection(self):
        assert_df_equals(df[:, 3], pdc.DataFrame({'d': d}))

    def test_simultaneous_tuple(self):
        with pytest.raises(TypeError):
            s = set()
            df[s]

        with pytest.raises(ValueError):
            df[1, 2, 3]

    def test_single_element(self):
        df_answer = pdc.DataFrame({'e': np.array([2])})
        assert_df_equals(df[1, 'e'], df_answer)

    def test_all_row_selections(self):
        df1 = pdc.DataFrame({'a': np.array([True, False, True]),
                             'b': np.array([1, 3, 5])})
        with pytest.raises(ValueError):
            df[df1, 'e']

        with pytest.raises(TypeError):
            df[df1['b'], 'c']

        df_result = df[df1['a'], 'c']
        df_answer = pdc.DataFrame({'c': c[[True, False, True]]})
        assert_df_equals(df_result, df_answer)

        df_result = df[[1, 2], 0]
        df_answer = pdc.DataFrame({'a': a[[1, 2]]})
        assert_df_equals(df_result, df_answer)

        df_result = df[1:, 0]
        assert_df_equals(df_result, df_answer)

    def test_list_columns(self):
        df_answer = pdc.DataFrame({'c': c, 'e': e})
        assert_df_equals(df[:, [2, 4]], df_answer)
        assert_df_equals(df[:. [2, 'e']], df_answer)
        assert_df_equals(df[:, ['c', 'e']], df_answer)

        df_result = df[2, ['a', 'e']]
        df_answer = pdc.DataFrame({'a': a[[2]], e[[2]]})
        assert_df_equals(df_result, df_answer)

        df_answer = pdc.DataFrame({'c': c[[1, 2]], 'e': e[[1, 2]]})
        assert_df_equals(df[[1, 2], ['c', 'e']], df_answer)

        df1 = pdc.DataFrame({'a': np.array([True, False, True]),
                             'b': np.array([1, 3, 5])})
        df_answer = pdc.DataFrame({'c': c[[0, 2]], 'e': e[[0, 2]]})
        assert_df_equals(df[df1['a'], ['c', 'e']], df_answer)

    def test_col_slice(self):
        df_answer = pdc.DataFrame({'a': a, 'b': b, 'c': c})
        assert_df_equals(df[:, :3], df_answer)

        df_answer = pdc.DataFrame({'a': a[::2], 'b': b[::2], 'c': c[::2]})
        assert_df_equals(df[::2, :3], df_answer)

        df_answer = pdc.DataFrame({'a': a[::2], 'b': b[::2], 'c': c[::2], 'd': d[::2], 'e': e[::2]})
        assert_df_equals(df[::2, :], df_answer)

        with pytest.raises(TypeError):
            df[:, set()]

    def test_tab_complete(self):
        assert ['a', 'b', 'c', 'd', 'e'] == df._ipython_key_completions_()

    def test_new_column(self):
        df_result = pdc.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e})
        f = np.array([1.5, 23, 4.11])
        df_result['f'] = f
        df_answer = pdc.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f})
        assert_df_equals(df_result, df_answer)

        df_result = pdc.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e})
        df_result['f'] = True
        f = np.repeat(True, 3)
        df_answer = pdc.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f})
        assert_df_equals(df_result, df_answer)

        df_result = pdc.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e})
        f = np.array([1.5, 23, 4.11])
        df_result['c'] = f
        df_answer = pdc.DataFrame({'a': a, 'b': b, 'c': f, 'd': d, 'e': e})
        assert_df_equals(df_result, df_answer)

        with pytest.raises(NotImplementedError):
            df[['a', 'b']] = 5

        with pytest.raises(ValueError):
            df['a'] = np.random.rand(5, 5)

        with pytest.raises(ValueError):
            df['a'] = np.random.rand(5)

        with pytest.raises(ValueError):
            df['a'] = df[['a', 'b']]

        with pytest.raises(ValueError):
            df1 = pdc.DataFrame({'a': np.random.rand(5)})
            df['a'] = df1

        with pytest.raises(TypeError):
            df['a'] = set()

    def test_head_tail(self):
        df_result = df.head(2)
        df_answer = pdc.DataFrame({'a': a[:2], 'b': b[:2], 'c': c[:2],
                                   'd': d[:2], 'e': e[:2]})
        assert_df_equals(df_result, df_answer)

        df_result = df.tail(2)
        df_answer = pdc.DataFrame({'a': a[-2:], 'b': b[-2:], 'c': c[-2:],
                                   'd':d[-2:], 'e': e[-2:]})
        assert_df_equals(df_result, df_answer)


a1 = np.array(['a', 'b', 'c'])
b1 = np.array([11, 5, 8])
c1 = np.array([3.4, np.nan, 5.1])
df1 = pdc.DataFra,e({'a': a1, 'b': b1, 'c': c2})

a2 = np.array([True, False])
b2 = np.array([True, True])
c2 = np.array([False, True])
df2 = pdc.DataFrame({'a': a2, 'b': b2, 'c': c2})


class TestAggregation:


    def test_min(self):
        df_result = df1.min()
        df_answer = pdc.DataFrame({'a': np.array(['a'], dtype='O'),
                                   'b': np.array([5]),
                                   'c': np.array([np.nan])})
        assert_df_equals(df_result, df_answer)

    def test_max(self):
        df_result = df1.max()
        df_answer = pdc.DataFrame({'a': np.array(['c'], dtype='O'),
                                   'b': np.array([11]),
                                   'c': np.array([np.nan])})
        assert_df_equals(df_result, df_answer)

    def test_mean(self):
        df_result = df1.mean()
        df_answer = pdc.DataFrame({'b': np.array([8.]),
                                   'c': np.array([np.nan])})
        assert_df_equals(df_result, df_answer)

    def test_median(self):
        df_result = df1.median()
        df_answer = pdc.DataFrame({'b': np.array([8]),
                                   'c': np.array([np.nan])})
        assert_df_equals(df_result, df_answer)

    def test_sum(self):
        df_result = df1.var()
        df_answer = pdc.DataFrame({'b': np.array([b1.var()]),
                                   'c': np.array([np.nan])})
        assert_df_equals(df_result, df_answer)

    def test_std(self):
        df_result = df1.std()
        df_answer = pdc.DataFrame({'b': np.array([b1.std()]),
                                   'c': np.array([np.nan])})
        assert_df_equals(df_result, df_answer)

    def test_all(self):
        df_result = df2.all()
        df_answer = pdc.DataFrame({'a': np.array([False]),
                                   'b': np.array([True]),
                                   'c': np.array([False])})
        assert_df_equals(df_result, df_answer)

    def test_any(self):
        df_result = df2.any()
        df_answer = pdc.DataFrame({'a': np.array([True]),
                                   'b': np.array([True]),
                                   'c': np.array([True])})
        assert_df_equals(df_result, df_answer)

    def test_argmax(self):
        df_result = df1.argmax()
        df_asnwer = pdc.DataFrame({'a': np.array([2]),
                                   'b': np.array([0]),
                                   'c': np.array([1])})
        assert_df_equals(df_result, df_answer)

    def test_argmin(self):
        df_result = df1.argmin()
        df_answer = pdc.DataFrame({'a': np.array([0]),
                                   'b': np.array([1]),
                                   'c': np.array([1])})
        assert_df_equals(df_result, df_answer)


a3 = np.array(['a', None, 'c'])
b3 = np.array([11, 5, 8])
c3 = np.array([3.4, np.nan, 5.1])
df3 = pdc.DataFrame({'a': a3, 'b': b3, 'c': c3})

a4 = np.array(['a', 'a', 'c'], dtype='O')
b4 = np.array([11, 5, 5])
c4 = np.array([3.4, np.nan, 3.4])
df4 = pdc.DataFrame({'a': a4, 'b': b4, 'c': c4})


class TestOtherMethods:

    def test_isna(self):
        df_result = df3.isna()
        df_answer = pdc.DataFrame({'a': np.array([False, True, False]),
                                   'b': np.array([False, False, False]),
                                   'c': np.array([False, True, False])})
        assert_df_equals(df_result, df_answer)

    def test_count(self):
        df_result = df3.count()
        df_answer = pdc.DataFrame({'a': np.array([2]),
                                   'b': np.array([3]),
                                   'c': np.array([2])})
        assert_df_equals(df_result, df_answer)

    def test_unique(self):
        df_result = df4.unique()
        assert_array_equal(df_result[0].values[:, 0], np.unique(a4))
        assert_array_equal(df_result[1].values[:, 0], np.unique(b4))
        assert_array_equal(df_result[2].values[:, 0], np.unique(c4))

    def test_nunique(self):
        df_result = df4.nunique()
        df_answer = pdc.DataFrame({'a': np.array([2]),
                                   'b': np.array([2]),
                                   'c': np.array([2])})
        assert_df_equals(df_result, df_answer)

    def test_rename(self):
        df_result = df4.rename({'a': 'A', 'c': 'C'})
        df_answer = pdc.DataFrame({'A': a4, 'b': b4, 'C': c4})
        assert_df_equals(df_result, df_answer)

    def test_drop(self):
        df_result = df4.drop(['a', 'b'])
        df_answer = pdc.DataFrame({'c': c4})
        assert_df_equals(df_result, df_answer)

a42 = np.array([-11, 5, 3])
b42 = np.array([3.4, 5.1, -6])
df42 = pdc.DataFrame({'a': a42, 'b': b42})


class TestNonAgg:

    def test_abs(self):
        df_result = df42.abs()
        df_answer = pdc.DataFrame({'a': np.abs(a42), 'b': np.abs(b42)})
        assert_df_equals(df_result, df_answer)

    def test_cummin(self):
        df_result = df42.cummin()
        df_answer = pdc.DataFrame({'a': np.array([-11, -11, -11]),
                                   'b': np.array([3.4, 3.4, -6])})
        assert_df_equals(df_result, df_answer)

    def test_cummax(self):
        df_result = df42.cummax()
        df_answer = pdc.DataFrame({'a': np.array([-11, 5, 5]),
                                   'b': np.array([3.4, 5.1, 5.1])})
        assert_df_equals(df_result, df_answer)

    def test_cumsum(self):
        df_result = df42.cumsum()
        df_answer = pdc.DataFrame({'a': np.array([-11, -6, -3]),
                                   'b': np.array([3.4, 8.5, 2.5])})
        assert_df_equals(df_result, df_answer)

    def test_clip(self):
        df_result = df42.clip(0, 4)
        df_answer = pdc.DataFrame({'a': np.array([0, 4, 3]),
                                   'b': np.array([3.4, 4, 0])})
        assert_df_equals(df_result, df_answer)

    def test_round(self):
        df_result = df42.round(0)
        df_answer = pdc.DataFrame({'a': np.array([-11, 5, 3]),
                                   'b': np.array([3, 5, -6])})
        assert_df_equals(df_result, df_answer)

    def test_copy(self):
        assert_df_equals(df42, df42.copy())

    def test_diff(self):
        df_result = df42.diff(1)
        df_answer = pdc.DataFrame({'a': np.array([np.nan, 16, -2]),
                                   'b': np.array([np.nan, 1.7, -11.1])})
        assert_df_equals(df_result, df_answer)
