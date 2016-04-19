from . import dataprocessing as dp
import json
import pandas as pd

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
