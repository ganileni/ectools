from unittest import TestCase
import numpy as np
from ectools.NWKR import NWKR
from multiprocessing import cpu_count


class TestNWKR(TestCase):
    def test_init(self):
        # check that it turns list into np.ndarray
        model = NWKR(bandwidth=[1, 1])
        self.assertTrue(isinstance(model.bw, np.ndarray))
        # check that if bw is single number it turns into float
        model = NWKR(bandwidth=[1])
        self.assertTrue(isinstance(model.bw, float))
        # kernel not callable or not in inlcuded list
        with self.assertRaises(TypeError):
            NWKR(kernel='abcdefgh')
        with self.assertRaises(TypeError):
            NWKR(kernel=1)

    def test_fit(self):
        X = np.array([[1, 1]])
        y = np.array([1])
        model = NWKR(kernel='norm', bandwidth=1, njobs=1)
        # X not right no. of dimensions
        with self.assertRaises(AssertionError):
            Xwrong = np.array([1])
            model.fit(Xwrong, y)
        # X not same dim as y
        with self.assertRaises(AssertionError):
            Xwrong = np.array([[1, 1], [1, 1]])
            model.fit(Xwrong, y)
        # all nan data
        with self.assertRaises(AssertionError):
            Xwrong = np.array([[np.nan]])
            model.fit(Xwrong, y)
        # check that bandwidth has same size as X
        with self.assertRaises(AssertionError):
            model_ = NWKR(bandwidth=[1, 1, 1])
            model_.fit(X, y)

    def test_predict(self):
        X = np.array([[1, 1]])
        y = np.array([1])
        model = NWKR(kernel='norm', bandwidth=1, njobs=1)
        # should predict 1, with std of 0
        model.fit(X, y)
        pred, std = model.predict(X)
        self.assertTrue(np.isclose(pred[0], float(1)))
        self.assertTrue(np.isclose(std[0], float(0)))
        # check that parallelization works
        if cpu_count() > 1:
            model_ = NWKR(kernel='norm', bandwidth=1, njobs=2)
            model_.fit(X,y)
            model_.predict(np.vstack([X] * 2))
