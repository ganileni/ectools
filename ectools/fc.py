import numpy as np
from numba import jit


def fitness_complexity_calculation(m, iterations, samples):
    """Returns fitness and complexity at different iteration stages for a given matrix m. m is intended to be of shape [n_countries,n_products], i.e. Fitness will be computed on the rows and Complexity on the columns of m.
    Note: this function handles all data cleaning procedures. The actual fixed point calculation is done by the function iterative_map(). iterative_map() was written for optimization with Numba, therefore its style is somewhat less than pythonic.

    Arguments:
        m: The binary matrix on which to compute the Fitness-Complexity algorithm. Must be np.ndarray of order 2
        iterations: How many iterations to do. Must be int.
        samples: How many samples to take while doing the iterations. Must be int. The sample will be equally spaced in time from 0 to `iterations`.

    Returns:
        fitness_value: A np.ndarray of size [`samples`,n_countries] containing the sampled values of Fitness for each country at different stages of iteration. For clarity, n_countries is the number of rows of `m`, the input matrix.
        fitness_normalization: A np.ndarray of size ['samples'] containing the normalization values of Fitness at different stages of iteration.
        complexity_value: A np.ndarray of size [`samples`,n_products] containing the sampled values of Complexity for each product at different stages of iteration. For clarity, n_products is the number of columns of `m`, the input matrix.
        complexity_normalization: A np.ndarray of size ['samples'] containing the normalization values of Complexity at different stages of iteration.
        sample_times: A np.ndarray of size ['samples'] containing the number of iterations for each of the samples output bu this function.

    """
    assert np.isnan(m).sum() == 0
    # if there are some products exported by nobody (!)
    noones_products = m.sum(axis=0) == 0
    noones_products_present = noones_products.sum()
    if noones_products_present:
        # remove columns corresponding to these products
        m = m[:, np.invert(noones_products)]
    # if there are some countries that export nothing (!)
    nontrade_countries = m.sum(axis=1) == 0
    nontrade_countries_present = nontrade_countries.sum()
    if nontrade_countries_present:
        # remove rows corresponding to such countries
        m = m[np.invert(nontrade_countries), :]
    # how big is the matrix
    num_countries, num_products = m.shape
    # where to take samples:
    if samples != 0:
        step = int(iterations / samples)
        sample_times = np.zeros(samples)
    else:
        step = iterations
        sample_times = np.array([0])
    # initialize fitness and complexity
    fitness = np.array(np.ones(num_countries), dtype='float128')
    complexity = np.array(np.ones(num_products), dtype='float128')
    # initialize sampling lists so that the algorithm doesn't have to do dynamic resizing of arrays
    fitness_value, fitness_normalization = \
        np.zeros([samples + 1, num_countries]), np.zeros(samples + 1)
    complexity_value, complexity_normalization = \
        np.zeros([samples + 1, num_products]), np.zeros(samples + 1)
    fitness_value, fitness_normalization, \
    complexity_value, complexity_normalization, sample_times = \
        iterative_map(m, iterations, step,
                      fitness, complexity,
                      fitness_value, fitness_normalization,
                      complexity_value, complexity_normalization,
                      sample_times)

    # values for countries that export nothing reinserted as nans:
    # (must come before reinsertion of problematic complexity values!)
    if nontrade_countries_present:
        # make empty arrays with adequate size
        fitness_value_complete = np.zeros([
            fitness_value.shape[0],
            len(nontrade_countries)])
        nan_vector = np.zeros(fitness_value_complete.shape[0])
        nan_vector[:] = np.nan
        # iteratively reinsert nontrading countries preserving original ordering
        count = 0
        for k, nontrading in enumerate(nontrade_countries):
            # if it's a trading country, put in its fitness
            if not nontrading:
                fitness_value_complete[:, k] = fitness_value[:, count]
                count += 1
            # otherwise put in a nan column
            else:
                fitness_value_complete[:, k] = nan_vector
        # substitute complete values in vector to be returned
        fitness_value = fitness_value_complete
    # values for products exported by nobody reinserted as nans:
    if noones_products_present:
        # make empty arrays with adequate size
        complexity_value_complete = np.zeros([
            complexity_value.shape[0],
            len(noones_products)])
        nan_vector = np.zeros([complexity_value_complete.shape[0]])
        nan_vector[:] = np.nan
        # iteratively reinsert noones products preserving original ordering
        count = 0
        for k, noones in enumerate(noones_products):
            # if it's not a problematic product
            if not noones:
                complexity_value_complete[:, k] = complexity_value[:, count]
                count += 1
            else:
                complexity_value_complete[:, k] = nan_vector
        # substitute complete values in vector to be returned
        complexity_value = complexity_value_complete
    return fitness_value, fitness_normalization, \
           complexity_value, complexity_normalization, sample_times


# numba jit doesn't support float128. commented until support someday.
# @jit(nopython = True)
def iterative_map(m, iterations, step,
                  fitness, complexity,
                  fitness_value, fitness_normalization,
                  complexity_value, complexity_normalization,
                  sample_times):
    samples_taken = 0
    converged = False
    for n in range(iterations):
        tmp_fitness = fitness
        tmp_complexity = complexity
        fitness = np.dot(m, tmp_complexity)
        complexity = 1 / (np.dot(1 / tmp_fitness, m))
        # normalize
        fitness = fitness / (fitness.sum() / len(fitness))
        complexity = complexity / (complexity.sum() / len(complexity))
        # check for nans in complexity
        # (nans appear there first, then in fitness)
        if np.isnan(complexity).sum() or converged:
            # remove all subsequent values from samples arrays
            fitness_value = fitness_value[:samples_taken + 1]
            complexity_value = complexity_value[:samples_taken + 1]
            fitness_normalization = fitness_normalization[:samples_taken + 1]
            complexity_normalization = complexity_normalization[:samples_taken + 1]
            sample_times = sample_times[:samples_taken + 1]
            # recover last non-nan fitness and complexity
            complexity = tmp_complexity
            fitness = tmp_fitness
            break
        # save iterations
        if n % step == 0:
            fitness_value[samples_taken] = fitness
            complexity_value[samples_taken] = complexity
            fitness_normalization[samples_taken] = fitness.sum()
            complexity_normalization[samples_taken] = complexity.sum()
            sample_times[samples_taken] = n
            samples_taken += 1
            # if : complexity[samples_taken] - complexity[samples_taken]
            #     converged = True
    fitness_value[-1] = fitness
    complexity_value[-1] = complexity
    fitness_normalization[-1] = fitness.sum()
    complexity_normalization[-1] = complexity.sum()
    sample_times[-1] = n
    return fitness_value, fitness_normalization, \
           complexity_value, complexity_normalization, \
           sample_times


def rank_tied(v, norm=True, alert=False):
    """Returns a vector containing TIED ranking of each element in `v`. NaN values are ignored by the ranking, and left as NaN in the output.
    Note: Tied ranking means that if `v` contains two or more elements that are equal, these elements will be assigned their average rank. E.g. if trhee elements with same value X are supposed to be arbitrarily assigned rank 4,5,6 in normal ranking, in tied ranking they will each get rank (4+5+6)/3=5.

    Arguments:
        v: np.ndarray of numeric type to be ranked
        norm: Whether to normalize ranking in [0,1]
        alert: print an alert to stdout if any ties are encountered.

    Returns:
        ranked_v: ranked vector.
    """
    # convert to numpy array
    if type(v) != type(np.zeros(1, dtype='float64')):
        v = np.array(v, dtype='float64')
    # initialize vector where ranked values go
    ranked_v = np.zeros(np.shape(v), dtype='float64')
    # nans in same places as v
    ranked_v[np.isnan(v)] = np.nan
    # copy on which you can make writes
    # np arrays are not immutable!
    copy_v = v.copy()
    # a number that is bigger than max(v) for sure
    max_v = np.nanmax(v) + 10e500
    ranked_v = fast_rank(ranked_v, copy_v, max_v, alert)
    # finally normalize to 1.
    if norm:
        ranked_v = ranked_v / np.nanmax(ranked_v)
    return (ranked_v)


@jit()
def fast_rank(ranked_v, copy_v, max_v, alert):
    """Helper function for rank_tied."""
    # number of ties
    ties = 0
    # number of cycles must be exactly len(v) minus n. of nans in v. one more cycle
    # will see all non-nan numbers be tied at 1
    for i in range(len(ranked_v) - np.sum(np.isnan(copy_v))):
        # skip `ties` cycles if there are ties
        if ties != 0:
            ties = ties - 1
            continue
        # find where the current minimum (minima) of copy_v are
        indices = np.where(copy_v == np.nanmin(copy_v))[0]
        # count the ties
        ties = len(indices)
        # if ties <=1, just rank
        if ties == 1:
            # write rank in `ranked _v`
            ranked_v[indices[0]] = i
            # remove the minima from `copy_v`
            copy_v[indices[0]] = max_v
            # ties is now 0, i.e. we took care
            # of all minima in `copy_v`
            ties = 0
        # if ties exist, then I need to assign
        # average of the next `ties` ranks to them
        else:
            if alert:
                print('found', str(ties), 'ties')
            average = 0
            # sum the next `ties` rankings:
            for k in range(ties):
                average = average + i + k
            average = average / ties
            # assign average ranking
            for n in indices:
                ranked_v[n] = average
                # remove minima from `max_v`
                copy_v[n] = max_v
            # now you're going to have to skip `ties`-1 cycles
            ties = ties - 1
    return ranked_v


def RCA(EXM):
    """Calculates RCA (Revealed Comparative Advantage) of export matrix EXM, according to Ballassa.

        Arguments:
            EXM: np.ndarray matrix of floats, of order 2. EXM is intended to be of shape [n_countries,n_products].

        Returns:
            RCA: np.ndarray matrix of floats, of shape [n_countries,n_products], containing the results of the calculation.
    """

    # totals by country, product and world
    totcountry = np.sum(EXM, axis=1)
    totproduct = np.sum(EXM, axis=0)
    totworld = np.sum(totproduct)
    RCA = np.copy(EXM)
    for i in range(len(RCA)):
        RCA[i, :] = RCA[i, :] / totcountry[i]
    # some countries have zero total exports (totcountry here),
    #  so the numerator of the RCA will be nan. let's set it to 0
    RCA[np.isnan(RCA)] = 0
    RCA = np.transpose(RCA)
    for i in range(len(RCA)):
        RCA[i, :] = RCA[i, :] / (totproduct[i] / totworld)
    # some products have zero total exports (totproducts here),
    #  so the denominator of the RCA will be nan.
    # let's set nans to 0
    RCA[np.isnan(RCA)] = 0
    RCA = np.transpose(RCA)
    return RCA


def logPRODY(RCA, gdppc):
    """Calculates logPRODY from an RCA matrix and a GDPpc vector, i.e. log gdppc per-product average on countries with weights equal to the RCA.

    Arguments:
        RCA: np.ndarray matrix of floats, of order 2. RCA is intended to be of shape [n_countries,n_products].
        gdppc: np.ndarray of floats, of order 2. Intended to be of shape [n_countries].

    Returns:
        logPRODY: np.ndarray floats, of shape [n_products], containing the results of the calculation.
    """
    RCA = np.array(RCA)
    gdppc = np.array(gdppc)
    # some countries have no gdppc data
    include = np.isfinite(gdppc)
    # initialize vector of zeros
    logPRODY = np.zeros(len(RCA[0, :]))
    # norm = will normalize RCAs by product
    # nb excluding countries with no gdppc data!
    # note the use of nansum, since some values might be nan in RCA matrix too!
    norm = np.nansum(RCA[include], axis=0)
    # calculate, excluding countries with no gdppc data
    for i in range(len(logPRODY)):
        logPRODY[i] = np.dot(RCA[include, i], np.log10(gdppc[include])) / norm[i]
    return logPRODY
