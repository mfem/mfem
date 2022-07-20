import argparse
import os
import json
from matplotlib.font_manager import json_load
from matplotlib.pyplot import axes
import numpy as np

from main import basic_example

class update_dual():

    def __init__(self, **kwargs):
        self.record = kwargs.get('record')
        self.record_path = kwargs.get('record_path')
        self.iteration = kwargs.get('iteration')

        config = kwargs.get('config')
        prob_config = config['prob_config']
        self.prob = basic_example(**prob_config)
        self.prob.x = record["Design iteration " + str(self.iteration)]['Design']
        self.prob.lam = record["Design iteration " + str(self.iteration)]['Lagrange multiplier']

        self.step_length = config['multiplier_update']['step_length']
        self.tolerance = config['multiplier_update']['tolerance']

    def __call__(self):
        self.update_multiplier()
        self.update_optimization_record()

    '''

        Finalize multiplier iteration

    '''
    def update_optimization_record(self):

        norm_reduced_grad = self.norm_of_reduced_gradient()
        if norm_reduced_grad < self.tolerance:
            record["Dual done"] = True
        else:
            record["Primal done"] = False
        record['Design iteration ' + str(self.iteration)]['Lagrange multiplier'] = self.prob.lam
        
        ## decrease sample size to begin next primal iteration
        # sample_size = record['Design iteration ' + str(self.iteration)]['Sample size']
        # record['Design iteration ' + str(self.iteration)]['Sample size'] = sample_size//2

        with open(self.record_path, 'w') as outfile:
            json.dump(record, outfile, indent=4)
    '''

        Compute the projected gradient descent update
        
            λ_{k+1} = λ_k - α G(x_k)

    '''
    def update_multiplier(self):
        self.prvs_lam = self.prob.lam
        self.prob.lam = self.prvs_lam - self.step_length * self.prob.G()

    '''
        Compute norm of reduced gradient || λ_{k+1} - λ_k || / α
    '''
    def norm_of_reduced_gradient(self):
        return abs(self.prob.lam - self.prvs_lam) / self.step_length

'''
    usage: dual_update.py [-h] [-c CONFIG_PATH] [-r RECORD_PATH]
                        [-i ITERATION]

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
    parser.add_argument('-o', '--output_path', type=str,
                        default='./', help="path of encore.yaml file")
     
    args = parser.parse_args()

    with open(args.config_path) as config_file:
        config = json.load(config_file)

    with open(args.record_path) as record_file:
        record = json.load(record_file)

    kwargs = {
        'config'      : config,
        'record'      : record,
        'record_path' : args.record_path,
        'iteration'   : args.iteration,
    }

    output_path = args.output_path
    encore = open(output_path + '/encore.yaml', "w")
    # encore2 = open(output_path + '../record.yaml', "w")
    
    primal_done = record["Primal done"]
    dual_done = record["Dual done"]
    if primal_done and dual_done:
        encore.write("is_done: True")
    elif primal_done:
        updater = update_dual(**kwargs)
        updater()
        encore.write("is_done: False")
    else:
        encore.write("is_done: False")