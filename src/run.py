import argparse
import datetime
import itertools
import json
import multiprocessing
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import re
import subprocess
import statistics
import sys
import torch
import tqdm
import warnings

import numpy as np
import pandas as pd

from collections import defaultdict
from multiprocessing import Pool

from data import DATASETS
from utils import find_best_epoch, check_path

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str, required=True)

    # Experimental setup
    parser.add_argument('--data', type=str, nargs='+')
    parser.add_argument('--gpu', type=int, nargs='+')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--trials', type=int)

    # Hyperparameters
    parser.add_argument('--batch-size', type=int, nargs='+')
    parser.add_argument('--dropout', type=float, nargs='+')

    return parser.parse_known_args()

def get_save_dir(exp):
    current_time = datetime.datetime.now()
    dir_name = current_time.strftime(f"./results/{exp}_%m_%d_%H_%M_%S")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def run_command(args):
    command, gpu_list = args
    if gpu_list:
        gpu_idx = int(multiprocessing.current_process().name.split('-')[-1]) - 1
        gpu = gpu_list[gpu_idx % len(gpu_list)]
        command += ['--gpu', str(gpu)]
    return subprocess.check_output(command)

def python_file(exp):
    return f'main/main_{exp}.py'

def save_args_to_json(args_dict, save_dir):
    json_file_path = os.path.join(save_dir, "0args.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)
    print(f"Arguments saved as JSON at: {json_file_path}")

def print_sum(dir):
    dataset_accuracies = {}

    # Loop through each file in the directory
    for filename in os.listdir(dir):
        if filename.count('_') == 3 or filename.count('_') == 4:
            filepath = os.path.join(dir, filename)
            with open(filepath, 'r') as file:
                data = json.load(file)
                dataset = data['data']
                best_test_acc = data['best_test_acc']

                if dataset not in dataset_accuracies:
                    dataset_accuracies[dataset] = []
                dataset_accuracies[dataset].append(best_test_acc)

    sorted_datasets = sorted(dataset_accuracies.keys())

    for dataset in sorted_datasets:
        accuracies = dataset_accuracies[dataset]
        average = statistics.mean(accuracies)
        if len(accuracies) > 1:
            stdev = statistics.stdev(accuracies)
        else:
            stdev = 0
        print(f"{dataset}, {average*100:.2f}%, {stdev*100:.2f}%")

def summarize(dir):
    dataset_accuracies = {}

    with open(os.path.join(dir, "0result.txt"), 'w') as result_file:

        # Loop through each file in the directory
        for filename in os.listdir(dir):
            if filename.count('_') == 3 or filename.count('_') == 4:
                filepath = os.path.join(dir, filename)
                with open(filepath, 'r') as file:
                    data = json.load(file)
                    dataset = data['data']
                    best_test_acc = data['best_test_acc']

                    if dataset not in dataset_accuracies:
                        dataset_accuracies[dataset] = []
                    dataset_accuracies[dataset].append(best_test_acc)

        sorted_datasets = sorted(dataset_accuracies.keys())

        for dataset in sorted_datasets:
            accuracies = dataset_accuracies[dataset]
            average = statistics.mean(accuracies)
            if len(accuracies) > 1:
                stdev = statistics.stdev(accuracies)
            else:
                stdev = 0
            print(f"Dataset: {dataset}, Average Best Test Accuracy: {average*100:.2f}%, Standard Deviation: {stdev*100:.2f}%", file=result_file)
    

def main():
    args, unknown = parse_args()
    args_dict = vars(args) 
    
    if unknown:
        print(f"Experiment: {args.exp}")
        print(unknown)
        it = iter(unknown)
        for arg in it:
            if arg.startswith('--'):
                key = arg[2:]
                try:
                    value = next(it)
                except StopIteration:
                    value = True 
                args_dict[key] = value

    if 'all' in [e.lower() for e in args.data]:
        args.data = DATASETS
    if torch.cuda.is_available() and not args.gpu:
        args.gpu = list(range(torch.cuda.device_count()))

    args.batch_size = sorted(args.batch_size)

    args_list = []

    save_dir = get_save_dir(args.exp)
    check_path(save_dir)
    save_args_to_json(args_dict, save_dir)

    file = python_file(args.exp)
    for data, seed in itertools.product(args.data, range(args.trials)):
        for b, d in itertools.product(args.batch_size, args.dropout):
            path = f"{save_dir}/{data}_{seed}_{b}_{d}"
            check_path(path)
            command = ['python', file, 
                       '--data', data, 
                       '--path', path,
                       '--fold', str(0),
                       '--batch-size', str(b),
                       '--dropout', str(d),
                       '--seed', str(seed)]
            args_list.append((command + unknown, args.gpu))

    out_list = []
    num_gpu = max(len(args.gpu), 1)
    with Pool(num_gpu * args.workers) as pool:
        for out in tqdm.tqdm(pool.imap_unordered(run_command, args_list), total=len(args_list)):
            out_list.append(out)
            summarize(save_dir)

    print(f"Result saved at: {save_dir}/0result.txt")

    print_sum(save_dir)

    print()
    print("----------------------------------------")
    print()

if __name__ == '__main__':
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
    main()
    