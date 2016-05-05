import json
import pandas as pd
from collections import OrderedDict
import inspect
import numpy as np
from . import io


class Printable:
    # Adapted from https://github.com/tdimiduk/yaml-serialize (updated for python 3)
    @property
    def _dict(self):
        dump_dict = OrderedDict()

        for var in inspect.signature(self.__init__).parameters:
            if getattr(self, var, None) is not None:
                item = getattr(self, var)
                if isinstance(item, np.ndarray) and item.ndim == 1:
                    item = list(item)
                dump_dict[var] = item

        return dump_dict

    def __repr__(self):
        keywpairs = ["{0}={1}".format(k[0], repr(k[1])) for k in self._dict.items()]
        return "{0}({1})".format(self.__class__.__name__, ", ".join(keywpairs))

    def __str__(self):
        return self.__repr__()


class Spectrum(Printable):
    """
    Represents a single spectrum. It contains a dataframe with the raw data
    as well as any metadata. The intention is to pass a dict to metadata so
    that arbitrary information can be stored.
    """

    def __init__(self, data, metadata):
        self.data = data
        self.metadata = metadata

    def plot(self):
        """
        Plots the data in the dataframe.
        (Will not work unless the object passed as data itself has a
        plot() method.)
        """

        self.data.plot()

    def to_json(self):
        """
        Outputs (as a string) a JSON representation of the data and metadata.
        """

        return (json.dumps(self.metadata)[:-1] +
                ", \"Spectrum\": " +
                self.data.to_json()+"}")

    # def __repr__(self):
    #    return json.dumps(self.to_json(), sort_keys=True,
    #            indent=4, separators=(',', ': '))

    def write_json(self, fp):
        """
        Writes a json representation of the spectrum object to a file.
        """
        f = open(fp, 'w')
        f.write(self.to_json())

    def swap_cols(self):
        """
        Swaps the order of the columns in the underlying dataframe.
        """
        cols = self.data.columns.tolist()
        reordered = [cols[1]] + [cols[0]]
        self.data = self.data[reordered]

    def add_to_metadata(self, new_dict):
        """
        Appends a dict to the existing metadata.
        """
        self.metadata = io.merge_dicts(self.metadata, new_dict)