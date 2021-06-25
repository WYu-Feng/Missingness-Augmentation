# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

from data_loader import data_loader
from ma_gain import MA_GAIN
from gain import GAIN
import os
import multiprocessing
from utils import rmse_loss

def main (args):
    data_name = args.data_name
    miss_rate = args.miss_rate
    alpha = args.alpha
    gain_parameters = {'batch_size': args.batch_size,
        'hint_rate': args.hint_rate,
        'alpha': args.alpha,
        'iterations': args.iterations} 
        
    # Load data and introduce missingness
    data_x, miss_data_x, data_m = data_loader(data_name, miss_rate)

    # Impute missing data
    # imputed_data = MA_GAIN(miss_data_x, gain_parameters)  #Start this line to implement the Missingness Augmentation

    imputed_data = GAIN(miss_data_x, gain_parameters)     #Start this line to implement the original method
    rmse = rmse_loss(data_x, imputed_data, data_m)
    print(data_name + ' dataset RMSE performance: ',rmse)


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        choices=['ionosphere', 'news, ', 'sonar', 'wine', 'winequality', 'iris','pendigits','abalone','avila','pendigits'],
        default='iris',
        type=str)
    parser.add_argument(
        '--miss_rate',
        help='missing data probability',
        default=0.5,
        type=float)
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch',
        default=64,
        type=int)
    parser.add_argument(
        '--hint_rate',
        help='hint probability',
        default=0.9,
        type=float)
    parser.add_argument(
        '--alpha',
        help='hyperparameter',
        default=100,
        type=float)   
    parser.add_argument(
        '--iterations',
        help='number of training interations',
        default=10000,
        type=int)

    args = parser.parse_args() 
    main(args)
