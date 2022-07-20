import argparse
import json
from matplotlib.font_manager import json_load
import numpy as np

from main import basic_example


'''
    Generate a gradient sample and save to file
'''
def save_obj_and_grad(**kwargs):
    config = kwargs.get('config')
    record_path = kwargs.get('record_path')
    iteration = kwargs.get('iteration')
    gradient_path = kwargs.get('gradient_path')

    prob_config = config['prob_config']
    prob = basic_example(**prob_config)

    with open(record_path) as record_file:
        record = json.load(record_file)

    prob.x = record["Design iteration " + str(iteration-1)]['Design']
    prob.lam = record["Design iteration "+str(iteration-1)]['Lagrange multiplier']

    multiplier_step_length = config['multiplier_update']['step_length']

    obj = np.asarray([prob.f()])
    lag = np.asarray([prob.L(multiplier_step_length)])
    grad = prob.gradL(multiplier_step_length)
    obj_lag_and_grad = np.concatenate((obj,lag,grad))
    np.savetxt(gradient_path, obj_lag_and_grad)


'''
    usage: primal_gradients.py [-h] [-c CONFIG_PATH] [-r RECORD_PATH]
                            [-g GRADIENT_PATH] [-i ITERATION] [-s SAMPLE]

    optional arguments:
    -h, --help            show this help message and exit
    -c CONFIG_PATH, --config_path CONFIG_PATH
                            config file
    -r RECORD_PATH, --record_path RECORD_PATH
                            optimization record file
    -g GRADIENT_PATH, --gradient_path GRADIENT_PATH
                            sample gradient filename
    -i ITERATION, --iteration ITERATION
                            iteration number
    -s SAMPLE, --sample SAMPLE
                            sample number
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str,
                        default='config.json', help="config file")
    parser.add_argument('-r', '--record_path', type=str,
                        default='record.json', help="optimization record file")
    parser.add_argument('-g', '--gradient_path', type=str,
                        default='sample_gradient.txt', help="sample gradient filename")
    parser.add_argument('-i', '--iteration', type=int,
                        default=1, help="iteration number")
    parser.add_argument('-s', '--sample', type=int,
                        default=1, help="sample number")
    args = parser.parse_args()

    with open(args.config_path) as config_file:
        config = json.load(config_file)

    kwargs = {
        'config'        : config,
        'gradient_path' : args.gradient_path,
        'record_path'   : args.record_path,
        'iteration'     : args.iteration
    }

    save_obj_and_grad(**kwargs)