from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize
from copy import deepcopy
from ectools.NWKR import NWKR
from itertools import combinations
from ectools.utilities import sort_n


def mae_error(pred, groundtruth):
    return np.nanmean(np.abs((pred - groundtruth)))
    """mean average error"""


def mse_error(pred, groundtruth):
    """mean root square error"""
    return np.sqrt(np.nanmean((pred - groundtruth) ** 2))

def mae_error_vect(pred, groundtruth):
    return np.abs(pred - groundtruth)
    """mean average error, element by element"""

def mse_error_vect(pred, groundtruth):
    """mean root square error, element by element"""
    return np.sqrt((pred - groundtruth) ** 2)


def mae_mse_mix(frac):
    assert 0 <= frac <= 1

    def mix(pred, groundtruth):
        return w * mae_error(pred, groundtruth) + (1 - w) * mse_error(pred, groundtruth)

    return mix


def mare_error(pred, groundtruth):
    """mean average relative error"""
    return np.nanmean(np.abs((pred - groundtruth) / groundtruth))


def msre_error(pred, groundtruth):
    """mean square relative error"""
    return np.nanmean(((pred - groundtruth) / groundtruth) ** 2) ** .5


class SPSb():
    """this class makes N-dimensional SPSb predictions given observed trajectories and velocities.
       basically a wrapper around the NWKR class that manages reshaping data arrays and renormalizing data"""

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
        assert len(set([(len(x.shape)) for x in points])) == 1 and len(points[0].shape) == 1
        # reshape data from long to wide
        X = np.vstack(points).T
        prediction, std = self.model.predict(X)
        # rescale standard devs
        std = std * np.sqrt(self.model.n)
        return prediction, std


def backtest_SPSb(trajectories, y_vel, bw, dt=5, kill_years=1, start_year_no=None):
    """this function backtests SPSb.
       the trajectories and velocities must be all observations, the function splits data to do the backtests.
       bw is the bandwidth for the NWKR
       dt is the time interval between predictions. should be the same dt used to compute velocities.
       kill_years is the number of initial years to be removed from the backtesting (because too little data results in poor predictions)
       start_year_no should be the year where your dataset begins. if not None, this will generate a list that describes predictions

       outputs: result, a namespace containing 4 attributes:
       pred - the prediction
       std - the uncertainty around the prediction
       groundtruth - the actual value measured
       dates - dates in a list that, for each row in the predictions array, specifies which years have been used to predict which interval in the format {start_year_used_data}-{end_year_used_data}: {start_year_prediction_time}-{end_year_prediction_time}
       """
    # decide whether to print the date list
    if start_year_no is not None:
        print_years = True
    else:
        print_years = False
    # result will be a namespace
    result = SimpleNamespace()
    # year where you should stop making predictions
    yearmax = trajectories[0].shape[0] - dt
    # nan filler for the skipped predictions
    filler = np.empty(trajectories[0].shape[1])
    filler[:] = np.nan
    result.pred = [filler] * kill_years
    result.std = [filler] * kill_years
    result.groundtruth = [filler] * kill_years
    # list of skipped predictions
    if print_years: result.dates = ['{}-{}: None'.format(start_year_no, start_year_no + dt + i - 1) for i in
                                    range(kill_years)]
    # iterate over all other possible predictions, backtesting
    for year in range(kill_years, yearmax):
        model = SPSb([x[:year] for x in trajectories], y_vel[:year], bw)
        pred, std = model.predict([x[year] for x in trajectories])
        result.pred.append(pred)
        result.std.append(std)
        result.groundtruth.append(y_vel[year + dt])
        # list of predictions
        if print_years: result.dates.append(
                '{}-{}: pred {}-{}'.format(*(start_year_no + np.array([0, year + dt - 1, year + dt, year + 2 * dt]))))
    # reshape data
    result.pred = np.vstack(result.pred)
    result.std = np.vstack(result.std)
    result.groundtruth = np.vstack(result.groundtruth)
    return result


class SPSb_optimizer:
    """tol set at None gives you the biggest precision. Empirically, a tol of 1e-5 results in big speedup (50 to 66%) and bandwidths that are ~3% different. again empirically, a difference of 3% in bandwidth results in a <10e-6 difference in MAE.
    so you might want to do huge sweeps with high tol, and then maybe look closer at interesting results with maximum precision."""

    def __init__(self, trajectories, y_vel,
                 dt, start_year_no, kill_years, which,
                 tol=None):
        self.trajectories = trajectories
        self.y_vel = y_vel
        self.dt = dt
        self.start_year_no = start_year_no
        self.kill_years = kill_years
        self.delay_y_vel = np.vstack([y_vel[:-dt]])
        self.set_errorfcn(which)
        self.tol = tol
        # parameters to be filled:
        self.bw = None
        self.SPSb_result = None
        self.SPSbV_result = None
        self.SPSb_weight = None
        self.errors = None

    def optimize(self, bw_guess=None, bw_bounds=None):
        # this should activate N-dimensional optimization
        if bw_guess == None: bw_guess = [.5] * len(self.trajectories)
        # first optimize bandwidths
        r = minimize(self.bw_error,
                     x0=bw_guess,
                     bounds=bw_bounds,
                     tol=self.tol,
                     )
        self.bw = r.x
        # then optimize for mae, mse, or a combination of the two
        r = minimize(self.weight_error, [.5], bounds=[[0, 1]], tol=self.tol)
        self.SPSb_weight = r.x[0]
        # then save results of calculations
        self.SPSbV_result = deepcopy(self.SPSb_result)  # this could be made faster by skipping copy
        self.SPSbV_result.pred = (self.SPSb_weight * self.SPSb_result.pred
                                  + (1 - self.SPSb_weight) * self.delay_y_vel)
        self.SPSbV_result.std[:] = None  # std is not available for SPSbV
        self.errors = SimpleNamespace()
        self.errors.SPSb = {
            'mse': mse_error(self.SPSb_result.pred, self.SPSb_result.groundtruth),
            'mae': mae_error(self.SPSb_result.pred, self.SPSb_result.groundtruth)
        }
        self.errors.SPSbV = {
            'mse': mse_error(self.SPSbV_result.pred, self.SPSbV_result.groundtruth),
            'mae': mae_error(self.SPSbV_result.pred, self.SPSbV_result.groundtruth)
        }

    def backtest(self, bw):
        self.SPSb_result = backtest_SPSb(trajectories=self.trajectories,
                                         y_vel=self.y_vel,
                                         bw=bw,
                                         dt=self.dt,
                                         kill_years=self.kill_years,
                                         start_year_no=self.start_year_no)

    def bw_error(self, bw):
        self.backtest(bw)
        return self.errorfcn(self.SPSb_result.pred, self.SPSb_result.groundtruth)

    def weight_error(self, w):
        return self.errorfcn(w * self.SPSb_result.pred
                             + (1 - w) * self.delay_y_vel,
                             self.SPSb_result.groundtruth)

    def set_errorfcn(self, which):
        if which == 'mae':
            self.errorfcn = mae_error
        elif which == 'mse':
            self.errorfcn = mse_error
        elif isinstance(which, number.Number):
            self.errorfcn = mae_mse_mix(which)
        elif which == None:
            self.errorfcn = lambda x: 0

class SPSb_optimizer_V_only:
    """tol set at None gives you the biggest precision. Empirically, a tol of 1e-5 results in big speedup (50 to 66%) and bandwidths that are ~3% different. again empirically, a difference of 3% in bandwidth results in a <10e-6 difference in MAE.
    so you might want to do huge sweeps with high tol, and then maybe look closer at interesting results with maximum precision."""

    def __init__(self, trajectories, y_vel,
                 dt, start_year_no, kill_years, which,
                 tol=None):
        self.trajectories = trajectories
        self.y_vel = y_vel
        self.dt = dt
        self.start_year_no = start_year_no
        self.kill_years = kill_years
        self.delay_y_vel = np.vstack([y_vel[:-dt]])
        self.set_errorfcn(which)
        self.tol = tol
        # parameters to be filled:
        self.bw = None
        self.SPSb_result = None
        self.SPSbV_result = None
        self.SPSb_weight = None
        self.errors = None

    def optimize(self, bw=None):
        self.bw = bw
        # then optimize for mae, mse, or a combination of the two
        r = minimize(self.weight_error, [.5], bounds=[[0, 1]], tol=self.tol)
        self.SPSb_weight = r.x[0]
        # then save results of calculations
        self.SPSbV_result = deepcopy(self.SPSb_result)  # this could be made faster by skipping copy
        self.SPSbV_result.pred = (self.SPSb_weight * self.SPSb_result.pred
                                  + (1 - self.SPSb_weight) * self.delay_y_vel)
        self.SPSbV_result.std[:] = None  # std is not available for SPSbV
        self.errors = SimpleNamespace()
        self.errors.SPSb = {
            'mse': mse_error(self.SPSb_result.pred, self.SPSb_result.groundtruth),
            'mae': mae_error(self.SPSb_result.pred, self.SPSb_result.groundtruth)
        }
        self.errors.SPSbV = {
            'mse': mse_error(self.SPSbV_result.pred, self.SPSbV_result.groundtruth),
            'mae': mae_error(self.SPSbV_result.pred, self.SPSbV_result.groundtruth)
        }

    def backtest(self, bw):
        self.SPSb_result = backtest_SPSb(trajectories=self.trajectories,
                                         y_vel=self.y_vel,
                                         bw=bw,
                                         dt=self.dt,
                                         kill_years=self.kill_years,
                                         start_year_no=self.start_year_no)

    def bw_error(self, bw):
        self.backtest(bw)
        return self.errorfcn(self.SPSb_result.pred, self.SPSb_result.groundtruth)

    def weight_error(self, w):
        return self.errorfcn(w * self.SPSb_result.pred
                             + (1 - w) * self.delay_y_vel,
                             self.SPSb_result.groundtruth)

    def set_errorfcn(self, which):
        if which == 'mae':
            self.errorfcn = mae_error
        elif which == 'mse':
            self.errorfcn = mse_error
        elif isinstance(which, number.Number):
            self.errorfcn = mae_mse_mix(which)
        elif which == None:
            self.errorfcn = lambda x: 0


def baselines(optimizer, n_iter=1000):
    """establish prediction baselines:
    'autocorrelation' is predicting with last observed velocity (it is known that GDP growth is autocorrelated)
    'random' is predicting with an observed velocity chosen at random from the distro of all observed velocities"""

    result = dict()
    # establish autocorrelation baseline
    result['autocorrelation'] = dict()
    result['autocorrelation']['mae'] = mae_error(optimizer.delay_y_vel, optimizer.SPSb_result.groundtruth)
    result['autocorrelation']['mse'] = mse_error(optimizer.delay_y_vel, optimizer.SPSb_result.groundtruth)

    # establish lower random baseline
    result['random'] = dict()
    shuffle_pred = optimizer.delay_y_vel.copy()
    shuffle_groundtruth = optimizer.SPSb_result.groundtruth.copy()
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


error_dict = {
    'mae': mae_error,
    'mse': mse_error
}

error_dict_vect = {
    'mae': mae_error_vect,
    'mse': mse_error_vect
}

def model_error(model, error='mae'):
    return error_dict[error](model.SPSb_result.pred,model.SPSb_result.groundtruth)

def model_error_vect(model, error='mae'):
    return error_dict_vect[error](model.SPSb_result.pred,model.SPSb_result.groundtruth)

def stack_error(model_list, error='mae', std_weigh=False):
    preds = np.array([model.SPSb_result.pred for model in model_list])
    if std_weigh:
        # weight predictions by 1/stds
        stds = np.array([1 / model.SPSb_result.std for model in model_list])
        preds = (preds * stds).sum(axis=0) / stds.sum(axis=0)
    else:
        preds = preds.mean(axis=0)
    groundtruth = model_list[0].SPSb_result.groundtruth
    return error_dict[error](preds, groundtruth)


def stack_error_improvement(model_list, error='mae', std_weigh=False):
    """takes a list of models (of class SPSb_optimizer). error is the error metric to use. Accepts 'mse' and 'mae'.
    error improvement obtained by stacking several models together."""
    #find minimum error
    minerror = np.min([model.errors.SPSb[error] for model in model_list])
    # find stacked error
    stack = stack_error(model_list, error=error, std_weigh=std_weigh)
    return (minerror - stack) / minerror


def error_correlation(model1, model2):
    """calculates the correlation of the errors of two models."""
    err1 = model1.SPSb_result.pred - model1.SPSb_result.groundtruth
    err2 = model2.SPSb_result.pred - model2.SPSb_result.groundtruth
    keep = np.isfinite(err1) * np.isfinite(err2)
    return np.corrcoef(x=err1[keep], y=err2[keep])[0, 1]


def stack_error_improvement_matrix(models, model_names, error='mae'):
    """takes as input a list of models (of class SPSb_optimizer) and a corresponding list of model names (as strings)
    outputs a symmetric matrix of error improvement obtained by stacking all possile combinations of models."""
    assert len(models) == len(model_names)
    n = len(models)
    # sort by increasing MSE of the model
    models, model_names = sort_n([models, model_names], key=lambda x: x.errors.SPSb[error])
    combs = combinations(list(range(n)), 2)
    result = np.empty((n, n))
    np.fill_diagonal(result, 0)
    for n1, n2 in combs:
        result[n1, n2] = result[n2, n1] = stack_error_improvement([models[n1], models[n2]], error=error)
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
