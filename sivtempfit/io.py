from . import dataprocessing as dp
import json
import pandas as pd
import os

def load_Spectrum(fp):
    """
    Creates a Spectrum object from a json file.
    Designed to read the output of the to_json() output from a Spectrum object
    """

    f = open(fp, 'r')
    loaded = json.load(f)
    data = pd.DataFrame(loaded['Spectrum'])
    metadata = {x : loaded[x] for x in loaded.keys() if x not in ['Spectrum']}
    return dp.Spectrum(data, metadata)

def get_example_data_file_path(filename, data_dir='exampledata'):
    # __file__ is the location of the source file currently in use (so
    # in this case io.py). We can use it as base path to construct
    # other paths from that should end up correct on other machines or
    # when the package is installed
    start = os.path.abspath(__file__)
    start_dir = os.path.dirname(start)
    # If you need to go up another directory (for example if you have
    # this function in your tests directory and your data is in the
    # package directory one level up) you can use
    # up_dir = os.path.split(start_dir)[0]
    data_dir = os.path.join(start_dir, data_dir)
    return os.path.join(start_dir, data_dir, filename)
