from os.path import dirname, exists
from ectools.utilities import pickleload, picklesave
from joblib import Parallel, delayed
from tqdm import tqdm


def _pw(fcn, iterable, n_jobs=1, progress=False, verbose=0, parallel_kwargs={}):
    """Helper function for `parallel_wrapper`.

        Args:
            fcn: the function to be calculated in parallel
            iterable: the iterable containing the argument for the function
            n_jobs: the number of CPU cores to use
            progress: whether to show a progressbar
            verbose: passed to the `verbose` keyword of joblib.Parallel
            parallel_kwargs: additional keyword arguments for joblib.Parallel

        Returns:
            results: a list containing the outputs of `fcn`
    """
    if progress:
        iterable = tqdm(iterable)
    with Parallel(n_jobs=n_jobs, verbose=verbose, **parallel_kwargs) as parallel:
        # calculate in parallel
        results = parallel(delayed(fcn)(_) for _ in iterable)
    return results


def parallel_wrapper(fcn, iterable, cache_to=None, n_jobs=1, progress=False, verbose=0, force_recompute=False,
                     parallel_kwargs={}):
    """A function that just wraps the most common use case of joblib. Parallelizes the calculation of `fcn` on the elements of `iterable`.

    Args:
        fcn: the function to be calculated in parallel
        iterable: the iterable containing the argument for the function
        cache_to: if it is a valid file path, and the file does not exist the calculation will be pickled to this path. If the file exists, the calculation will be skipped, and the function will read the result from this path.
        n_jobs: the number of CPU cores to use
        progress: whether to show a progressbar using tqdm
        verbose: passed to the `verbose` keyword of joblib.Parallel
        force recompute: makes the wrapper ignore `cache_to`, and recomputes even if the file exists.
        parallel_kwargs: additional keyword arguments for joblib.Parallel

    Returns:
            results: a list containing the outputs of `fcn`
    """
    # if caching is active
    if cache_to is not None:
        # check directory exists
        directory_name = dirname(cache_to)
        if directory_name:
            if not exists(directory_name):
                raise FileNotFoundError('The directory specified for the caching of the computation does not exist.')
        # do computation
        if force_recompute or not exists(cache_to):
            results = _pw(fcn=fcn, iterable=iterable, n_jobs=n_jobs, progress=progress, verbose=verbose, parallel_kwargs=parallel_kwargs)
            # save result
            picklesave(results, cache_to)
        # if file exists:
        else:
            results = pickleload(cache_to)
    else:
        results = _pw(fcn=fcn, iterable=iterable, n_jobs=n_jobs, progress=progress, verbose=verbose, parallel_kwargs=parallel_kwargs)
    return results
