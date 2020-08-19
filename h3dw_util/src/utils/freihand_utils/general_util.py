from __future__ import print_function, unicode_literals
import os
import numpy as np
import json


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.float32):
            return float(obj)

        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def json_dump(file_name, data, pretty_format=False, overwrite=True, verbose=False):
    msg = 'File does exists and should not be overwritten: %s' % file_name
    assert not os.path.exists(file_name) or overwrite, msg

    with open(file_name, 'w') as fo:
        if pretty_format:
            json.dump(data, fo, cls=NumpyEncoder, sort_keys=True, indent=4)
        else:
            json.dump(data, fo, cls=NumpyEncoder)

    if verbose:
        print('Dumped %d entries to file %s' % (len(data), file_name))


def json_load(file_name):
    with open(file_name, 'r') as fi:
        data = json.load(fi)
    return data

