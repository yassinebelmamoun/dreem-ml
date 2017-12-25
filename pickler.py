import pandas as pd
from pathlib import Path 
from errno import ENOENT
import os

def get_or_create_pickle(file, force = False):
    # Get pickle from csv file name.
    # PS: We can manually force a re-pickling if file changed

    path_file = 'dataset/' + file
    file_pickle = file + '.p'
    path_pickle = 'dataset/' + file_pickle

    if Path('dataset/' + file_pickle).is_file() and not(force):
        print('The file *** ' + file + ' *** is already pickled')
    else:
        print('Read csv file : ' + file)
        df = pd.read_csv(path_file, index_col=0) #nrows=100
        print('Pickling: ' + file)
        df.to_pickle(path_pickle)
        print('Pickler done for: ' + file)
    try:
        return pd.read_pickle(path_pickle)
    except:
        print('Error')