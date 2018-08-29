from numba import jit
import numpy as np
import numbers
from ectools.parallelization import parallel_wrapper


# kernel definitions
@jit(nopython=True)
def norm_kernel(distances):
    """gaussian kernel"""
    # no gamma because bandwidth is calculated elsewhere
    return np.sqrt(np.pi / 2) * np.exp(-distances ** 2)


# list of kernels
_kernel_list = {
    'norm': norm_kernel
}


@jit(nopython=True)
def _predict(X, X_t, y_t, bw, kernel):
    """helper function for NWKR. It is fully compiled in Numba, and it vectorizes all that is vectorizable to improve speed.

    Args:
        X: the observations to predict from
        X_t: training dataset observations
        y_t: training dataset predictions
        bw: bandwidth(s) - will perform broadcasting
        kernel: a callable that evaluates kernel densities from euclidean distances

    Returns:
        prediction_out: predictions for observations X
        std_out: standard deviations on predictions for observations X
    """
    # initialize arrays for output
    prediction_out = np.empty(X.shape[0])
    std_out = np.empty(X.shape[0])
    # iterate over each row (observation requiring prediction)
    for k in range(X.shape[0]):
        # normal NWKR algorithm
        distances = np.sqrt((((X_t - X[k, :]) / bw) ** 2).sum(axis=1))
        weights = kernel(distances)
        weights = weights / np.sum(weights)  # normalize
        prediction_out[k] = np.sum(weights * y_t)
        std_out[k] = (np.sum(weights * (y_t ** 2)) - prediction_out[k] ** 2) ** .5
    return prediction_out, std_out


class NWKR():
    """Nadaraya-Watson kernel regression in the style of Scikit-learn.

    The difference with scikit-learn's is that it supports NaN values both in the fit() and predict() methods: it will output np.nan accordingly. Another difference is that it automatically computes the standard deviations of the estimated prediction distribution. See

    for the definition of NWKR and

    for the standard deviations calculation.
    Optimized for speed with Numba.

    Args:
        kernel: must be a string with the name of one of the included kernels (see NWKR._kernel_list) or callable.
        bandwidth: the bandwidth that scale distances in the euclidean space of observations. must be single float or iterable of floats (for different bandwidths along different dimensions).
        njobs: number of CPU cores to use for computation. njobs>1 requires joblib.
    """

    def __init__(self, kernel='norm', bandwidth=1, njobs=1):
        # bandwidth can also be a np.array, with multiple bw for each dimension
        self.bw = bandwidth
        self.njobs = njobs
        # if it's a number or a 1-d numpy array:
        if (isinstance(bandwidth, numbers.Number)
                or (isinstance(bandwidth, np.ndarray) and bandwidth.size == 1)):
            self.multiple_bw = False
            if isinstance(bandwidth, np.ndarray):
                # if it's 1-d numpy array, turn into a float
                self.bw = self.bw[0]
        else:
            self.multiple_bw = True
            # turn into np.array, needed later for broadcasting
            self.bw = np.array(self.bw)
        if kernel in _kernel_list:
            self.kernel = _kernel_list[kernel]
        else:
            if hasattr(kernel,'__call__'):
                self.kernel = kernel
            else:
                raise TypeError('kernel must be callable or one of included kernels (check NWKR._kernel_list)')

    def fit(self, X, y):
        """Fit using observations X and outcomes y.

        Args:
            X: a np.ndarray of floats of dimension 2
            y: a np.ndarray of floats with the same number of elements as the rows of X

        Returns:
            None

        """
        assert X.ndim == 2
        assert X.shape[0] == y.size
        if self.multiple_bw:
            # check that bandwidth has same no. of elements as no. of features in X
            assert len(self.bw.shape) == 1 and self.bw.size == X.shape[1]
        # only keep complete samples
        keep = np.invert(np.isnan(X).sum(axis=1) > 0) & np.isfinite(y)
        assert keep.sum()>0
        self.X_t = X[keep]
        self.y_t = y[keep]
        self.n = self.X_t.shape[0]

    def predict(self, X):
        """Predict outcomes from observations X, and also calculate standard deviations for the predictions.

        Args:
            X: a np.ndarray of floats, of dimension 2.

        Returns:
            predictions: a np.ndarray with a prediction for each row in X
            predictions_std: standard deviations on the predictions
        """
        assert X.ndim == 2
        assert X.shape[1] == self.X_t.shape[1]
        if self.njobs == 1:
            predictions, stds = _predict(X, self.X_t, self.y_t, self.bw, self.kernel)
        else:  # parallelize
            def parallel_helper(X):
                return _predict(X, self.X_t, self.y_t, self.bw, self.kernel)

            result = parallel_wrapper(fcn=parallel_helper,
                                      iterable=np.array_split(X, self.njobs, axis=0),
                                      n_jobs=self.njobs, progress=False)
            predictions, stds = zip(*result)
            predictions = np.hstack(predictions)
            stds = np.hstack(stds)
        return predictions, stds

