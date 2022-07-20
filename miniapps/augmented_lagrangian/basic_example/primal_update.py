import argparse
import json
from matplotlib.font_manager import json_load
from matplotlib.pyplot import axes
import numpy as np
from main import basic_example

class update_primal():

    def __init__(self, **kwargs):
        self.record = kwargs.get('record')
        self.record_path = kwargs.get('record_path')
        self.gradient_path_list = kwargs.get('gradient_path_list')
        self.iteration = kwargs.get('iteration')

        config = kwargs.get('config')
        prob_config = config['prob_config']
        self.prob = basic_example(**prob_config)
        self.prob.x = np.array(self.record["Design iteration "+str(self.iteration-1)]['Design'])
        self.prob.lam = record["Design iteration "+str(self.iteration-1)]['Lagrange multiplier']

        self.step_length = config['design_update']['step_length']
        self.norm_test_factor = config['design_update']['norm_test_factor']
        self.tolerance = config['design_update']['tolerance']

    def __call__(self):
        self.compute_mean_and_variance()
        self.update_design()
        self.update_sample_size()
        self.update_optimization_record()

    '''

        Estimate the mean of the sample gradients,
        
            ∇F ≈ 1/N Σ_i ∇f_i ,

        estimate the variance of the sample gradients,

            σ² ≈ 1/(N-1) || ∇f_i - ∇F ||² / N ,

        and estimate the norm of the sample gradients,

            || ∇F ||² .

    '''
    def compute_mean_and_variance(self):

        self.sample_size = sum(1 for _ in open(self.gradient_path_list))
        with open(self.gradient_path_list) as path_list:
            file_path = path_list.readline().rstrip()
            nx = sum(1 for _ in open(file_path)) - 2

        obj = np.zeros(self.sample_size)
        lag = np.zeros(self.sample_size)
        grad = np.zeros((nx,self.sample_size))
        with open(self.gradient_path_list) as path_list:
            for sample, file_path in enumerate(path_list):
                obj_lag_and_grad = np.loadtxt(file_path.rstrip())
                obj[sample] = obj_lag_and_grad[0]
                lag[sample] = obj_lag_and_grad[1]
                grad[:,sample] = obj_lag_and_grad[2:]

        self.mean_obj = np.mean(obj)
        self.mean_lag = np.mean(lag)
        self.mean_grad = np.mean(grad, axis=1)
        norm2_of_mean = np.linalg.norm(self.mean_grad)**2
        mean_of_norm2 = np.mean(np.linalg.norm(grad, axis=0)**2)
        norm_variance = (mean_of_norm2 - norm2_of_mean) / (self.sample_size - 1)
        self.ratio = norm_variance / ( self.norm_test_factor**2 * norm2_of_mean + 1e-12)

    '''

        Check the norm test and update sample size

    '''
    def update_sample_size(self):
        if self.ratio > 1:
            self.sample_size = int(self.ratio * self.sample_size)

    def update_optimization_record(self):

        norm_reduced_grad = self.norm_of_reduced_gradient()
        if norm_reduced_grad < self.tolerance:
            record["Primal done"] = True
        else:
            record["Dual done"] = False
        record["Sample size"] = self.sample_size
        record["Design iteration "+str(self.iteration-1)]["Ratio"]      = self.ratio
        record["Design iteration "+str(self.iteration-1)]["Objective"]  = self.mean_obj
        record["Design iteration "+str(self.iteration-1)]["Lagrangian"] = self.mean_lag
        record["Design iteration "+str(self.iteration-1)]["norm_reduced_grad"] = norm_reduced_grad
        record["Design iteration "+str(self.iteration)] = {
            "Sample size"         : self.sample_size,
            "Ratio"               : 0.0, # Computed at the next iteration
            "Objective"           : 0.0, # Computed at the next iteration
            "Lagrangian"          : 0.0, # Computed at the next iteration
            "Constraint"          : self.prob.G(),
            "Design"              : list(self.prob.x),
            "Lagrange multiplier" : self.prob.lam
        }

        with open(self.record_path, 'w') as outfile:
            json.dump(record, outfile, indent=4)

    '''

        Compute the projected gradient descent update
        
            x_{k+1} = P( x_k - α ∇F(x_k) )

    '''
    def update_design(self):
        self.prvs_x = self.prob.x
        self.prob.x = self.prvs_x - self.step_length * self.mean_grad

    '''
        Compute norm of reduced gradient || x_{k+1} - x_k || / α
    '''
    def norm_of_reduced_gradient(self):
        return np.linalg.norm(self.prob.x - self.prvs_x) / self.step_length

'''
    usage: primal_update.py [-h] [-c CONFIG_PATH] [-r RECORD_PATH]
                            [-i ITERATION] [-gl GRADIENT_PATH_LIST]

    optional arguments:
    -h, --help            show this help message and exit
    -c CONFIG_PATH, --config_path CONFIG_PATH
                            config file
    -r RECORD_PATH, --record_path RECORD_PATH
                            optimization record file
    -i ITERATION, --iteration ITERATION
                            iteration number
    -gl GRADIENT_PATH_LIST, --gradient_path_list GRADIENT_PATH_LIST
                            sample gradient path list
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str,
                        default='config.json', help="config file")
    parser.add_argument('-r', '--record_path', type=str,
                        default='record.json', help="optimization record file")
    parser.add_argument('-i', '--iteration', type=int,
                        default=1, help="iteration number")
    parser.add_argument('-gl', '--gradient_path_list', type=str,
                        default='sample_gradient_list.txt', help="sample gradient path list")

    args = parser.parse_args()

    with open(args.config_path) as config_file:
        config = json.load(config_file)

    with open(args.record_path) as record_file:
        record = json.load(record_file)

    kwargs = {
        'config'             : config,
        'record'             : record,
        'record_path'        : args.record_path,
        'gradient_path_list' : args.gradient_path_list,
        'iteration'          : args.iteration,
    }

    update_obj = update_primal(**kwargs)
    update_obj()