from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize
from ectools.NWKR import NWKR
from itertools import combinations
from ectools.utilities import sort_n


# TODO -- implement masks in stack error evaluation

def error_vect(pred, groundtruth):
    return pred - groundtruth
    """error, elementwise"""


def abs_error_vect(pred, groundtruth):
    return np.abs(pred - groundtruth)
    """absolute error, elementwise"""


def mare_error_vect(pred, groundtruth):
    """mean average relative error, elementwise"""
    return np.abs((pred - groundtruth) / groundtruth)


def mae_error(pred, groundtruth):
    return np.nanmean(abs_error_vect(pred, groundtruth))
    """mean average error"""


def mse_error(pred, groundtruth):
    """mean root square error"""
    return np.sqrt(np.nanmean(np.power(error_vect(pred, groundtruth), 2)))


def mae_mse_mix(frac):
    assert 0 <= frac <= 1

    def mix(pred, groundtruth):
        return frac * mae_error(pred, groundtruth) + (1 - frac) * mse_error(pred, groundtruth)

    return mix


def mare_error(pred, groundtruth):
    """mean average relative error"""
    return np.nanmean(mare_error_vect(pred, groundtruth))


def msre_error(pred, groundtruth):
    """mean square relative error"""
    return np.sqrt(np.nanmean(np.power(mare_error_vect(pred, groundtruth), 2)))


error_dict = {
    'mae': mae_error,
    'mse': mse_error,
    'mare': mare_error,
    'msre': msre_error,
}

error_dict_vect = {
    'abs': abs_error_vect,
    'err': error_vect,
    'mare': mare_error_vect,
}


class SPSb():
    """This class makes N-dimensional SPSb predictions given observed trajectories and velocities.
       It is basically a wrapper around the NWKR class that manages reshaping data arrays and renormalizing data.

       Arguments:
           trajectories: a list of numpy arrays, one for each indicator used to make a prediction. Each numpy array should be of shape [n_years,n_countries]. They will be used as features by NWKR.
           y_vel: the values to be used as prediction variable y by NWKR.
           bandwidth: the bandwidth that scale distances in the euclidean space of observations. must be single float or iterable of floats of order 1 (i.e. a vector, for different bandwidths along different dimensions).

       Returns:
           an SPSb object.
       """

    def __init__(self, trajectories, y_vel, bandwidth=.6):
        self.trajectories = trajectories
        self.y_vel = y_vel
        assert len(set([x.shape for x in self.trajectories] + [self.y_vel.shape])) == 1
        self.bw = bandwidth
        self.model = NWKR(kernel='norm', bandwidth=self.bw)
        # reshape data from wide to long format
        X = np.vstack([x.flatten() for x in self.trajectories]).T
        y = self.y_vel.flatten()
        self.model.fit(X, y)

    def predict(self, points):
        """Predict using the SPSb model.

        Arguments:
            points: a list of coordinate vectors, one for each prediction to be made. Each coordinate vector must have the same number of elements as there are dimensions in the SPSb model fitted at initialization.

        Returns:
            predictions: SPSb predictions
            std: the expected error of the prediction
        """
        assert len(set([(len(x.shape)) for x in points])) == 1 and len(points[0].shape) == 1
        # reshape data from long to wide
        X = np.vstack(points).T
        prediction, std = self.model.predict(X)
        # rescale standard devs
        std = std * np.sqrt(self.model.n)
        return prediction, std


def format_date(start_year_no, dt, year, msg=''):
    """Helper function for backtest_SPSb()"""
    return '{}-{}: pred {}-{}{}'.format(
            *([start_year_no + 0,
               start_year_no + year + dt - 1,
               start_year_no + year + dt,
               start_year_no + year + 2 * dt]),
            ''.join([' (', msg, ')']) if msg else '')


class SPSbResult(SimpleNamespace):
    """
    A class to hold the results of one SPSb computation. The results are accessed as attributes of the object.

    Arguments:
        pred: np.ndarray containing the predictions. Must be of shape [n_years,n_countries].
        std: np.ndarray containing the standard deviations (expected errors of prediction). Must be of shape [n_years,n_countries].
        groundtruth: np.ndarray containing the ground truth. Must be of shape [n_years,n_countries].
        dt: int containing the time interval of the predictions. Used internally.

    Returns:
        self: an SPSbResult object.

    Attributes:
        pred, std, groundtruth: see Arguments.
        errors: a dict containing the prediction errors. Lazily computed on first access.
        delay_y: ground truth delayed by self.dt years. Lazily computed on first access.
    """
    def __init__(self, pred, groundtruth, std, dt):
        super().__init__()
        self.pred = pred
        self.groundtruth = groundtruth
        self.std = std
        self.dt = dt


    @property
    def errors(self):
        """Lazily computed delayed model errors"""
        try:
            return self._errors
        except AttributeError:
            self._compute_error()
            return self._errors

    def _compute_error(self):
        self._errors = dict()
        for e in ['mae', 'mse']:
            self._errors[e] = error_dict[e](self.pred, self.groundtruth)

    @property
    def delay_y(self):
        """Lazily computed delayed groundtruth vector"""
        try:
            return self._delay_y
        except AttributeError:
            self._add_delay_y()
            return self._delay_y

    def _add_delay_y(self):
        self._delay_y = np.vstack([self.groundtruth[:-self.dt]] + [np.zeros(self.groundtruth.shape[1])] * self.dt)


def backtest_SPSb(trajectories, y, bw, dt=5, kill_years=1, yearmax=None, start_year_no=None, skip_no_groundtruth=False):
    """This function backtests SPSb.

       Arguments:
           trajectories: A list of numpy arrays, one for each indicator used to make a prediction. Each numpy array should be of shape [n_years,n_countries]. They will be used as features by NWKR. The trajectories and velocities must be all available observations, the function splits data automatically to do the backtests.
           bw: The bandwidth for the NWKR. must be single float or iterable of floats of order 1 (i.e. a vector, for different bandwidths along different dimensions).
           dt: The time interval between predictions. Should be the same dt used to compute velocities.
           kill_years: The number of initial years to be removed from the backtesting (because using too little data results in poor predictions).
           start_year_no: Should be the year where your dataset begins. if not None, this will generate a list that describes the time intervals for predictions.
           no_groundtruth_pred: whether to make a prediction on years where there is no groundtruth to compare with.

       Returns:
           result: A namespace containing 4 attributes:
               pred - The out-of-sample predictions. Will contain NaNs where the prediction has not been made.
               std - The out-of-sample uncertainty around the prediction
               groundtruth - The actual values measured, useful to compute error
               dates - Time interval of predictions. A list that, for each row in the predictions array, specifies which years have been used to predict which interval in the format {start_year_used_data}-{end_year_used_data}: {start_year_prediction_time}-{end_year_prediction_time}
       """
    assert trajectories[0].shape[0] > kill_years >= 1
    if yearmax is not None:
        assert 0 < yearmax < trajectories[0].shape[0]
    else:
        yearmax = trajectories[0].shape[0]
    # decide whether to print the date list
    if start_year_no is not None:
        print_years = True
    else:
        print_years = False
    # nan filler for the skipped predictions
    filler = np.empty(trajectories[0].shape[1])
    filler[:] = np.nan
    result = SPSbResult(pred=[filler] * kill_years,
                        std=[filler] * kill_years,
                        groundtruth=[filler] * kill_years,
                        dt=dt
                        )
    # list of skipped predictions
    if print_years: result.dates = [format_date(start_year_no, dt, i, msg='killed') for i in range(kill_years)]
    # iterate over all other possible predictions, backtesting
    for year in range(kill_years, yearmax):
        try:
            groundtruth = y[year + dt]
            groundtruth_data_exists = np.any(np.isfinite(groundtruth))
        except IndexError:
            groundtruth = filler
            groundtruth_data_exists = False
        if skip_no_groundtruth and not groundtruth_data_exists:
            result.pred.append(filler)
            result.std.append(filler)
            result.groundtruth.append(filler)
            if print_years: result.dates.append(
                    format_date(start_year_no, dt, year, msg='skipped: no ground truth data'))
            continue
        model = SPSb([x[:year] for x in trajectories], y[:year], bw)
        pred, std = model.predict([x[year] for x in trajectories])
        result.pred.append(pred)
        result.std.append(std)
        result.groundtruth.append(groundtruth)
        # list of predictions
        if print_years: result.dates.append(
                format_date(start_year_no, dt, year, msg='' if groundtruth_data_exists else 'no groundtruth data'))
    # add the remaining dt+no_pred years to the array so that its size matches that of y
    times = trajectories[0].shape[0] - yearmax
    result.pred += [filler] * times
    result.std += [filler] * times
    result.groundtruth += [filler] * times
    if print_years: result.dates += [format_date(start_year_no, dt, i, msg='killed') for i in
                                     range(yearmax, yearmax + times)]
    # reshape data
    result.pred = np.vstack(result.pred)
    result.std = np.vstack(result.std)
    result.groundtruth = np.vstack(result.groundtruth)
    return result


def optimize_SPSb_bandwidth(trajectories, y, dt, kill_years, error=error_dict['mae'], bw_guess=None, bw_bounds=None,
                            tol=None):
    """Helper function for SPSbBandwidthOptimizer, can also be used on its own..

    Optimizes SPSb bandwidth along all directions on train data i.e. in-sample.

    The out-of-sample prediction made by backtest_SPSb() avoids making a prediction on the first `kill_years` years, so you can minimize error with respect to bandwidth on these years without any information from the test set leaking into the (bandwidth) metaparameter choice.

    Arguments:
        trajectories: same argument passed to backtest_SPSb() call for which we are optimizing bandwidth.
        y: same argument passed to backtest_SPSb() call for which we are optimizing bandwidth.
        dt: same argument passed to backtest_SPSb() call for which we are optimizing bandwidth.
        kill_years: same argument passed to backtest_SPSb() call for which we are optimizing bandwidth. This is used to divide the data in test set and trainig set.
        error: which cost function to use in order to optimize bandwidth.
        bw_guess: a guess for the optimizer to start the search. Must be iterable with a float for each element in trajectories (ie for each direction). See documentation of scipy.optimize.minimize. If 'auto' will either use the scipy.optimize.minimize()'s best guess or the mean of bw_bounds.
        bw_bounds: boundaries for search of the minimum. Must be iterable with a tuple of floats for each element in trajectories (ie for each direction). See documentation of scipy.optimize.minimize. If 'auto', will compute the distance distribution of tha data along each direction, and set the bounds for the minimum search to the 1st and 95th percentile of this distribution.
        tol: tolerance for the optimizer. See documentation of scipy.optimize.minimize. tol set at None gives you the highest precision. Empirically, a tol of 1e-5 results in big speedups (50 to 66%) and bandwidths that are ~3% different. again empirically, a difference of 3% in bandwidth results in a <10e-6 difference in MAE. so you might want to do huge sweeps with high tol, and then maybe look closer at interesting results with maximum precision.

    Returns:
        r: a vector of optimal bandwidths, one for each element in trajectories.
    """
    # this will activate N-dimensional optimization
    if bw_guess is None: bw_guess = [.5] * len(trajectories)

    def cost_fcn(bw):
        result = backtest_SPSb(trajectories=trajectories, y=y, dt=dt, bw=bw, kill_years=1, yearmax=kill_years,
                               skip_no_groundtruth=True)
        return error(result.pred, result.groundtruth)

    r = minimize(cost_fcn,
                 x0=bw_guess,
                 bounds=bw_bounds,
                 tol=tol,
                 )
    return r.x


def optimize_velocity_SPSb_weight(trajectories, y, delay_y, dt, kill_years, error=error_dict['mae']):
    """Helper function for SPSb_optimizer, can also be used on its own.

    Optimizes out-of-sample the weight between SPSb and starting velocity to get the optimal velocity SPSb model.

    Arguments:
        trajectories: same argument passed to backtest_SPSb() call for which we are optimizing the weight.
        y: same argument passed to backtest_SPSb() call for which we are optimizing the weight.
        dt: same argument passed to backtest_SPSb() call for which we are optimizing the weight.
        kill_years: same argument passed to backtest_SPSb() call for which we are optimizing the weight. This is used to divide the data in test set and trainig set.
        error: which cost function to use in order to optimize the weight.

    Returns:
        r: optimal weight for SPSb predictions (ie to obtain velocity SPSb: multiply the prediction by r and the delayed velocity by 1-r, and finally sum).
    """

    result = backtest_SPSb(trajectories=trajectories, y=y, dt=dt, kill_years=1, yearmax=kill_years,
                           no_pred=trajectories[0].shape[0] - dt - kill_years)

    def cost_fcn(ratio): return error(ratio * result.pred + (1 - ratio) * delay_y, result.groundtruth)

    r = minimize(cost_fcn, x0=.5, bounds=[0, 1], tol=1e-3)
    return r.x


def distances_distribution(g, sample=None):
    """
    Computes all the distances between each two pairs of elements in float vector `g`. ie if g has N elements, will compute N*(N-1)/2 distances and return them in a vector.

    Arguments:
        g: np.ndarray of floats
        sample: number of elements to sample at random from g to compute the distances. Use if g is very big. Defaults to None, which uses all of g.

    Returns:
        distances: a np.ndarray of floats containing the N*(N-1)/2 distances computed.
    """
    g = g[np.isfinite(g)].ravel()
    if sample is not None:
        g = np.random.choice(g, sample, replace=True if sample >= g.size else False)
    return np.abs(np.triu(np.transpose([g]) - g)[np.triu_indices(len(g), k=1)])


class SPSbBandwidthOptimizer:
    """
    Optimizes the bandwidth along all directions of an SPSb model. The optimization is done in-sample (ie on train data), so that future information doesn't leak into the training of the model and backtesting is rigorous. See SPSb.optimize_SPSb_bandwidth() for more information.

    Arguments:
        trajectories: same argument passed to backtest_SPSb() call for which we are optimizing bandwidth.
        y: same argument passed to backtest_SPSb() call for which we are optimizing bandwidth.
        dt: same argument passed to backtest_SPSb() call for which we are optimizing bandwidth.
        kill_years: same argument passed to backtest_SPSb() call for which we are optimizing bandwidth. This is used to divide the data in test set and trainig set.
        start_year_no: int referencing the starting year of the dataset. It's used to write down strings that describe the timeframes of every column of the model's output.

    Returns:
        An SPSbBandwidthOptimizer object.
    """

    def __init__(self, trajectories, y,
                 dt, start_year_no, kill_years):
        self.trajectories = trajectories
        self.y = y
        self.dt = dt
        self.start_year_no = start_year_no
        self.kill_years = kill_years
        self.is_optimized = False
        # parameters to be filled:
        self.bw = None
        self.SPSb_result = None

    def fit(self, bw_guess='auto', bw_bounds='distances', tol=None, error=error_dict['mae'], save=False):
        """
        Optimizes the bandwidth.

        Arguments:
            error: which cost function to use in order to optimize bandwidth.
            bw_guess: a guess for the optimizer to start the search. Must be iterable with a float for each element in trajectories (ie for each direction). See documentation of scipy.optimize.minimize. If 'auto' will either use the scipy.optimize.minimize()'s best guess or the mean of bw_bounds.
            bw_bounds: boundaries for search of the minimum. Must be iterable with a tuple of floats for each element in trajectories (ie for each direction). See documentation of scipy.optimize.minimize. If 'auto', will compute the distance distribution of tha data along each direction, and set the bounds for the minimum search to the 1st and 95th percentile of this distribution.
            tol: tolerance for the optimizer. See documentation of scipy.optimize.minimize. tol set at None gives you the highest precision.
            save: whether to save the result and metaparameters in the object.

        Returns:
            bw: a list of floats containing the bandwidths, one for each direction of the model.
        """
        if bw_bounds == 'distances':
            bw_bounds = [np.percentile(distances_distribution(t.ravel()), [1, 95]) for t in self.trajectories]
        elif bw_bounds == 'auto':
            bw_bounds = None
        if bw_guess == 'auto':
            if bw_bounds is None:
                bw_guess = None
            else:
                bw_guess = [np.mean(x) for x in bw_bounds]
        if isinstance(error, str):
            error = error_dict[error]
        if save:
            self.error = error
            self.bw_guess = bw_guess
            self.bw_bounds = bw_bounds
            self.tol = tol
        self.bw = optimize_SPSb_bandwidth(self.trajectories, self.y, self.dt, self.kill_years, error=self.error,
                                          bw_guess=bw_guess, bw_bounds=bw_bounds, tol=tol)
        self.is_optimized = True
        return self.bw

    def predict(self, skip_no_groundtruth=True):
        """
        Computes the results of the model with the optimized bandwidths.

        Arguments:
            skip_no_groundtruth: whether to skip computing the result of the model on years where the ground truth is missing, to reduce amount of computation if you are just testing.

        Returns:
            SPSb_result: an SPSbResult object
        """
        if not self.is_optimized: raise Exception(
                'Bandwidth not optimized yet. Call SPSb_bw_optimizer.optimize_bw() before!')
        # compute with optimal bw, then save results of calculations
        self.SPSb_result = backtest_SPSb(trajectories=self.trajectories,
                                         y=self.y,
                                         bw=self.bw,
                                         dt=self.dt,
                                         kill_years=self.kill_years,
                                         yearmax=None,
                                         skip_no_groundtruth=skip_no_groundtruth,
                                         start_year_no=self.start_year_no)
        # compute error and save
        return self.SPSb_result

    def fit_predict(self, bw_guess='auto', bw_bounds='distances', tol=None, error=error_dict['mae'], save=False,
                    skip_no_groundtruth=True):
        self.fit(bw_guess=bw_guess, bw_bounds=bw_bounds, tol=tol, error=error, save=save)
        return self.predict(skip_no_groundtruth=skip_no_groundtruth)


class VelocitySPSbOptimizer:
    def __init__(self, bandwidth_optimizer: SPSbBandwidthOptimizer):
        self.opt = bandwidth_optimizer
        self.opt.delay_y()
        self.ratio = None

    def fit(self, error=error_dict['mae']):
        self.ratio = optimize_velocity_SPSb_weight(self.opt.trajectories, self.opt.y, self.delay_y, self.opt.dt,
                                                   self.opt.kill_years, error=error)

    def predict(self):
        self.SPSb_result = SPSbResult(pred = self.ratio * self.opt.SPSb_result.pred + (1 - self.ratio) * self.opt.SPSb_result.delay_y,
                                      groundtruth= self.opt.SPSb_result.groundtruth,
                                      std=None,
                                      dt=self.opt.dt)
        return self.SPSb_result

    def fit_predict(self, error=error_dict['mae']):
        self.fit(error)
        return self.predict()


def baselines(model: SPSbResult, n_iter=1000):
    """Establishes prediction baselines:

    'autocorrelation' consists in predicting with last observed velocity (it is known that GDP growth is autocorrelated),
    'random' consists in predicting with an observed velocity chosen at random from the distribution of all observed velocities.

    Arguments:
        model: an SPSbResult object, which contains the SPSb predictions (i.e. after calling the SPSbBandwidthOptimizer.predict() method).
        n_iter: the amount of iterations to compute the random baseline.

    Returns:
        result: a nested dict containing the baseline results.
    """

    result = dict()
    # establish autocorrelation baseline
    result['autocorrelation'] = dict()
    result['autocorrelation']['mae'] = mae_error(model.delay_y, model.groundtruth)
    result['autocorrelation']['mse'] = mse_error(model.delay_y, model.groundtruth)

    # establish lower random baseline
    result['random'] = dict()
    shuffle_pred = model.delay_y.copy()
    shuffle_groundtruth = model.groundtruth.copy()
    keep = np.isfinite(shuffle_pred) & np.isfinite(shuffle_groundtruth)
    shuffle_pred = shuffle_pred[keep]
    shuffle_groundtruth = shuffle_groundtruth[keep]
    shuffle_mse, shuffle_mae = [], []
    for j in range(n_iter):
        np.random.shuffle(shuffle_pred)
        shuffle_mae.append(mae_error(shuffle_pred, shuffle_groundtruth))
        shuffle_mse.append(mse_error(shuffle_pred, shuffle_groundtruth))
    result['random']['mae'] = np.mean(shuffle_mae)
    result['random']['mse'] = np.mean(shuffle_mse)
    result['random']['mae_std'] = np.std(shuffle_mae)
    result['random']['mse_std'] = np.std(shuffle_mse)
    return result


def model_error(result: SPSbResult, error='mae'):
    """Computes the error of a model.

    Arguments:
        model: an SPSbResult object, which contains the SPSb predictions (i.e. after calling the SPSbBandwidthOptimizer.predict() method).
        error: a string which indicates the error function to use. See  SPSb.error_dict.

    Returns:
        model_error: a float, the error of the model.
    """
    return error_dict[error](result.pred, result.groundtruth)


def model_error_vect(result: SPSbResult, error='abs'):
    """Computes the errors of a model, elementwise (i.e. prediction by prediction, not averaged).

    Arguments:
        result: an SPSbResult object, which contains the SPSb predictions (i.e. after calling the SPSbBandwidthOptimizer.predict() method).
        error: a string which indicates the error function to use. See  SPSb.error_dict_vect.

    Returns:
        model_error_vect: a np.ndarray containing the model error elementwise.
        """
    return error_dict_vect[error](result.pred, result.groundtruth)


def stack_error(result_list, error='mae', std_weigh=False):
    """
    Compute the error of a stacked model (i.e. where each prediction is the weighted average of the predictions of separate models).

    Arguments:
        result_list: a list of SPSbResult objects, each containing the SPSb predictions.
        error: a string which indicates the error function to use. See  SPSb.error_dict_vect.
        std_weigh: if False, all models will be weighed equally. If True, each model prediction will have weight inversely proportional to its standard deviation (ie the prediction error).

    Returns:
         error: The error of the stacked model.
    """
    preds = np.array([model.pred for model in result_list])
    if std_weigh:
        # weight predictions by 1/stds
        stds = np.array([1 / model.std for model in result_list])
        preds = (preds * stds).sum(axis=0) / stds.sum(axis=0)
    else:
        preds = preds.mean(axis=0)
    groundtruth = result_list[0].groundtruth
    return error_dict[error](preds, groundtruth)


def stack_error_improvement(result_list, error='mae', std_weigh=False):
    """ Computes error improvement obtained by stacking several models together.

    Defining minerror=min([error(x) for x in model_list]) and stack as the error of the stacked model, Stack Error Improvement is calculated as:
            (minerror - stack) / minerror

    Arguments:
        result_list: a list of SPSbResult objects, each containing the SPSb predictions.
        error: a string which indicates the error function to use. See  SPSb.error_dict_vect.
        std_weigh: if False, all models will be weighed equally. If True, each model prediction will have weight inversely proportional to its standard deviation (ie the prediction error).

    Returns:
        stack_error_improvement: Stack Error Improvement
    """
    # find minimum error
    minerror = np.min([res.errors[error] for res in result_list])
    # find stacked error
    stack = stack_error(result_list, error=error, std_weigh=std_weigh)
    return (minerror - stack) / minerror


def error_correlation(result1, result2):
    """Calculates the correlation of the errors of two models.

    Arguments:
        result1, result2: an SPSbResult object, which contains the SPSb predictions (i.e. after calling the SPSbBandwidthOptimizer.predict() method).

    Returns:
        corr: correlation of errors of the two models.
    """
    err1 = error_vect(result1.pred, result1.groundtruth)
    err2 = error_vect(result2.pred, result2.groundtruth)
    keep = np.isfinite(err1) * np.isfinite(err2)
    return np.corrcoef(x=err1[keep], y=err2[keep])[0, 1]


def stack_error_improvement_matrix(result_list, model_names, error='mae'):
    """Computes the stacked error improvement matrix, which for each element (i,j) contains the pairwise stacked error improvement of model_list i and j. Returns as pd.DataFrame.

    Arguments:
        result_list: a list of SPSbBandwidthOptimizer objects, each containing the SPSb predictions.
        model_names: a list of string with the names of the models. Will be used as column and row names in the returned pd.DataFrame.
        error: a string which indicates the error function to use. See  SPSb.error_dict_vect.

    Returns:
        stack_error_improvement_matrix: the stacked error improvement matrix, as a pd.DataFrame.
    """
    assert len(result_list) == len(model_names)
    n = len(result_list)
    # sort by increasing MSE of the model
    result_list, model_names = sort_n([result_list, model_names], key=lambda x: x.errors[error])
    combs = combinations(list(range(n)), 2)
    result = np.empty((n, n))
    np.fill_diagonal(result, 0)
    for n1, n2 in combs:
        result[n1, n2] = result[n2, n1] = stack_error_improvement([result_list[n1], result_list[n2]], error=error)
        if result[n1, n2] > 1:
            print(model_names[n1], model_names[n2])
    result = pd.DataFrame(result)
    result.columns = result.index = model_names
    return result


def best_model_from_stack_matrix(M):
    """finds best model by detecting the largest positive submatrix in M. returns observable list ordered by increasing model error."""
    included = [M.columns[0]]
    for obs in M.columns[1:]:
        # check if obs is orthogonal to at least one of the the observables in the included set
        for inc in included:
            if M.loc[obs, inc] > 0:
                include = True
        # now if it is parallel to at least one of the observable, reverse your choice of including it.
        for inc in included:
            if M.loc[obs, inc] <= 0:
                include = False
        if include: included.append(obs)
    return included


def best_model_from_stack_matrix_v2(M):
    """finds best model. returns observable list ordered by increasing model error.
    difference with V1: observables are sorted by increasing order of contribution
    to model improvement. it doesn't change results (doublecheck on several datasets)"""
    included = [M.columns[0]]
    # sort observables in order of increasing contribution to improvement
    _, sorted_obs = sort_n([M.loc[obs, :], M.columns[1:]], reverse=True)
    for obs in sorted_obs:
        # check if obs is orthogonal to at least one of the the observables in the included set
        for inc in included:
            if M.loc[obs, inc] > 0:
                include = True
        # now if it is parallel to at least one of the observable, reverse your choice of including it.
        for inc in included:
            if M.loc[obs, inc] <= 0:
                include = False
        if include: included.append(obs)
        return included
