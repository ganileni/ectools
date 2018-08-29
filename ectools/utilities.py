import pickle


def pickleload(path):
    """wrap pickle with context manager"""
    with open(path, 'rb') as file:
        loaded = pickle.load(file)
    return loaded


def picklesave(obj, path):
    """wrap pickle with context manager"""
    with open(path, 'wb') as file:
        pickle.dump(obj, file)


def unzip(zipped):
    """inverse operation of zip"""
    return zip(*zipped)


def sort_n(lists, key=None, reverse=False):
    """sort a list of lists by applying `key` to the first list"""
    lists = [list(x) for x in lists]
    if key is None:
        return unzip(sorted(zip(*lists), reverse=reverse))
    else:
        return unzip(sorted(zip(*lists), key=lambda x: key(x[0]), reverse=reverse))
