import argparse
import json
import numpy as np
import pandas as pd

from main import basic_example

class initialize():

    def __init__(self, **kwargs):
        self.config = kwargs.get('config')
        self.record_path = kwargs.get('record_path')
        self.prob_config = config['prob_config']
        self.prob = basic_example(**self.prob_config)

    def __call__(self):
        self.generate_initial_guess()
        self.intitialize_optimization_record()

    '''
        Generate an initial guesses and write to file
    '''
    def generate_initial_guess(self):
        # NOTE: Could use the problem class to generate a "random" initial guess
        pass

    '''
        Initialize the optimization record
    '''
    def intitialize_optimization_record(self):
        
        sample_size = self.config['design_update']['initial_sample_size']
        record = {
            "Primal done" : False,
            "Dual done" : False,
            "Sample size" : sample_size,
            "Design iteration 0" : {
                "Sample size"         : sample_size,
                "Ratio"               : 0.0, # Entry is filled in later
                "Objective"           : 0.0, # Entry is filled in later
                "Lagrangian"          : 0.0, # Entry is filled in later
                "Constraint"          : self.prob.G(),
                "Design"              : list(self.prob_config['x0']),
                "Lagrange multiplier" : self.prob_config['lam0'],
            }
        }

        # NOTE: Could also store as a Pandas dataframe (probably better)
        # df = pd.data(record)

        with open(self.record_path, 'w') as outfile:
            json.dump(record, outfile, indent=4)


'''
    usage: initialize.py [-h] [-c CONFIG_PATH] [-r RECORD_PATH] [-i ITERATION]

    optional arguments:
    -h, --help            show this help message and exit
    -c CONFIG_PATH, --config_path CONFIG_PATH
                            config file
    -r RECORD_PATH, --record_path RECORD_PATH
                            optimization record file
    -i ITERATION, --iteration ITERATION
                            iteration number
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str,
                        default='config.json', help="config file")
    parser.add_argument('-r', '--record_path', type=str,
                        default='record.json', help="optimization record file")
    parser.add_argument('-i', '--iteration', type=int,
                        default=1, help="iteration number")
    args = parser.parse_args()

    if args.iteration == 1:

        with open(args.config_path) as config_file:
            config = json.load(config_file)

        kwargs = {
            'config'      : config,
            'record_path' : args.record_path,
        }

        initializer = initialize(**kwargs)
        initializer()