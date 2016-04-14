import json


class spectrum():
    """
    Represents a single spectrum. It contains a dataframe with the raw data
    as well as the metadata. The intention is to pass a dict to metadata so
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

        return json.dumps(self.metadata)[:-1]
        + ", \"Spectrum\": "
        + self.data.to_json()+"}"

    def write_json(self, fp):
        pass
