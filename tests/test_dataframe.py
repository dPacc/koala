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
