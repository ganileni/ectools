# ectools

**ectools** is a collection of the Python tools used to do the calculations in these Economic Complexity papers:

- [*Angelini, O., Cristelli, M., Zaccaria, A., & Pietronero, L. (2017). The complex dynamics of products and its asymptotic properties. PLoS ONE, 12(5).*](https://doi.org/10.1371/journal.pone.0177360)
   
- [*Angelini, O., & Di Matteo, T. (2018). Complexity of products: the effect of data regularisation.*](http://arxiv.org/abs/1808.08249)

I plan to release most or ideally all of the code publicly, and will incrementally add it to this repository. For the moment the code is simple enough that the docstrings should cover it.
Please contact me for any queries regarding its use or any bugs that you may find.

At the moment, the repository contains:
- **NWKR**: An implementation of the Nadaraya-Watson Kernel Regression in the style of scikit-learn.
Compared to that found in scikit-learn, it supports `np.nan` values in the input and calculates the standard deviation of the predictions.
Implemented in numba for speed
- **parallelization**: A wrapper around joblib for its most common use case.
Mainly included as a dependency of NWKR.
- **utilities**: Utility functions included as a dependency.

## Motivation

The code is published to help with reproducibility, and in support of the open source philosophy.

## Code Example
```
import numpy as np
from ectools.NWKR import NWKR
X = np.array([[1,1],[1,1]])
y = np.array([1,1])
model = NWKR()
model.fit(X,y)
prediction, prediction_std = model.predict(X)
print(prediction, prediction_std)
```

## Installation

The code has been tested with python 3.6.
Just pull the code with

`git clone git@github.com:ganileni/ectools.git`

and install the requirements:

`cd ectools`

`pip install -r requirements.txt`

if the tests work correctly:

`cd tests/`

`nosetests`

you're good to go.

You can install the package with

`python setup.py install`

or only symlink the files in your python installation with

`python setup.py develop`

## Contributors

Just me.

## License

vanilla MIT license. Check out [LICENSE](LICENSE).
