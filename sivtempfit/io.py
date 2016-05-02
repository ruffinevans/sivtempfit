from . import dataprocessing as dp
import json
import pandas as pd
import os
import warnings

def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

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

def import_horiba_multi(fp, metadata = None, metadatalist = None,
                        keylabel = "Temperature"):
    """
    Imports csv files created by the Horiba iHR550 spectrometer from multiple
    spectra with temperature data included as the first line.

    usage: import_horiba_multi("some_file.csv")
    returns: list of Spectrum objects

    Also accepts as optional arguments:
    metadata : metadata to append to each Spectrum
    metadatalist : list of metadata with the same length as the number of
                   spectra. Each element in this list is included with the
                   corresponding spectrum object.
    """
    
    # Generate metadata from file
    file_meta = {"File creation time" : os.path.getctime(fp),
                "Original file path" : fp}
    
    temperatures_raw = pd.read_csv(fp, nrows = 1, header = None).T
    temperatures = temperatures_raw[1:].values.flatten()
    
    temperature_dicts = [{"Temperature" : i} for i in temperatures]
    
    cols = temperatures.size

    spectra = pd.read_csv(fp, header = 1)
    spectra_x_key = spectra.keys()[0]
    spectra_y_keys = spectra.keys()[1:]
    
    spectra_out = [dp.Spectrum] * cols
    
    for i in range(cols):
        try:
            spectra_out[i] = dp.Spectrum(spectra[[spectra_x_key, spectra_y_keys[i]]],
                                        merge_dicts(file_meta, temperature_dicts[i],
                                                    metadata, metadatalist[i]))
        except (IndexError, TypeError):
            # Assume the problem is that metadatalist is not passed
            # or is out of elements
            try:
                spectra_out[i] = dp.Spectrum(spectra[[spectra_x_key, spectra_y_keys[i]]],
                                        merge_dicts(file_meta, temperature_dicts[i],
                                                    metadata))
            except (IndexError, TypeError):
                # Now, assume that both metadata and metadatalist are not there
                # Currently fails if metadatalist is passed and metadata is not.
                if not(metadata == None and metadatalist == None):
                    warnings.warn("Metadata not parsed correctly. Please check inputs.")
                spectra_out[i] = dp.Spectrum(spectra[[spectra_x_key, spectra_y_keys[i]]],
                                        merge_dicts(file_meta, temperature_dicts[i]))

    # Swap columns to get y-first. Makes it compatible with the rest of the
    # code. Very unfortunate convention.
    [x.swap_cols() for x in spectra_out]
    
    return spectra_out