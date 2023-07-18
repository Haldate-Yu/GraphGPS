import os
import csv
import torch
import random
import numpy as np
from pynvml import *
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.utils.io import (
    dict_list_to_json,
    dict_list_to_tb,
    dict_to_json,
    json_to_dict_list,
    makedirs_rm_exist,
    string_to_python,
)


def is_seed(s):
    try:
        int(s)
        return True
    except Exception:
        return False


def is_split(s):
    if s in ['train', 'val', 'test']:
        return True
    else:
        return False


def agg_dict_list(dict_list):
    """
    Aggregate a list of dictionaries: mean + std
    Args:
        dict_list: list of dictionaries

    """
    dict_agg = {'epoch': dict_list[0]['epoch']}
    for key in dict_list[0]:
        if key != 'epoch':
            value = np.array([dict[key] for dict in dict_list])
            dict_agg[key] = np.mean(value).round(cfg.round)
            dict_agg['{}_std'.format(key)] = np.std(value).round(cfg.round)
    return dict_agg


def num_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def num_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_gpu_utilization(device_index):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device_index)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")
    return info.used // 1024 ** 2


def results_to_file(args, test_acc, test_std,
                    val_acc, val_std,
                    total_time, total_time_std,
                    avg_time, avg_time_std):
    if not os.path.exists('./results/{}'.format(args.dataset)):
        print("=" * 20)
        print("Creating Results File !!!")

        os.makedirs('./results/{}'.format(args.dataset))

    filename = "./results/{}/result_batchsize{}_layers{}.csv".format(
        args.dataset, args.batch_size, args.num_encoder_layers)

    headerList = ["Method", "N_Heads", "Batch_Size",
                  "Encoder_Layers", "Hidden_Dims",
                  "Model_Params", "Memory_Usage(MB)",
                  "::::::::",
                  "test_acc", "test_std",
                  "val_acc", "val_std",
                  "total_time", "total_time_std",
                  "avg_time", "avg_time_std"]

    with open(filename, "a+") as f:

        # reader = csv.reader(f)
        # row1 = next(reader)
        f.seek(0)
        header = f.read(6)
        if header != "Method":
            dw = csv.DictWriter(f, delimiter=',',
                                fieldnames=headerList)
            dw.writeheader()

        line = "{}, {}, {}, {}, {}, {}, {}, :::::::::, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n".format(
            args.model_type, args.nhead, args.batch_size,
            args.num_encoder_layers, args.model_dim,
            args.total_params, args.memory_usage,
            test_acc, test_std,
            val_acc, val_std,
            total_time, total_time_std,
            avg_time, avg_time_std
        )
        f.write(line)


def agg_runs_to_csv(cfg, dir, metric_best='auto'):
    if not os.path.exists('./results/{}'.format(cfg.dataset.name)):
        print("=" * 20)
        print("Creating Results File !!!")

        os.makedirs('./results/{}'.format(cfg.dataset.name))

    filename = "./results/{}/result_batchsize{}_layers{}.csv".format(
        cfg.dataset.name, cfg.train.batch_size, cfg.gt.layers)

    # processing results
    results_best = {'train': None, 'val': None, 'test': None}
    for seed in os.listdir(dir):
        if is_seed(seed):
            dir_seed = os.path.join(dir, seed)

            split = 'val'
            if split in os.listdir(dir_seed):
                dir_split = os.path.join(dir_seed, split)
                fname_stats = os.path.join(dir_split, 'stats.json')
                stats_list = json_to_dict_list(fname_stats)
                if metric_best == 'auto':
                    metric = 'auc' if 'auc' in stats_list[0] else 'accuracy'
                else:
                    metric = metric_best
                performance_np = np.array(  # noqa
                    [stats[metric] for stats in stats_list])
                best_epoch = \
                    stats_list[
                        eval("performance_np.{}()".format(cfg.metric_agg))][
                        'epoch']

            for split in os.listdir(dir_seed):
                if is_split(split):
                    dir_split = os.path.join(dir_seed, split)
                    fname_stats = os.path.join(dir_split, 'stats.json')
                    stats_list = json_to_dict_list(fname_stats)
                    stats_best = [
                        stats for stats in stats_list
                        if stats['epoch'] == best_epoch
                    ][0]

                    if results_best[split] is None:
                        results_best[split] = [stats_best]
                    else:
                        results_best[split] += [stats_best]
    results_best = {k: v
                    for k, v in results_best.items()
                    if v is not None}  # rm None
    for key in results_best:
        results_best[key] = agg_dict_list(results_best[key])

    # test best result
    test_best_results = results_best['test']

    headerList = ["Method", "N_Heads", "Batch_Size",
                  "Encoder_Layers", "Hidden_Dims",
                  "Model_Params", "Memory_Usage(MB)",
                  "::::::::",
                  "test_acc", "test_std",

                  "total_time", "total_time_std",
                  "avg_time", "avg_time_std",
                  "test_time", "test_time_std"]

    with open(filename, "a+") as f:

        # reader = csv.reader(f)
        # row1 = next(reader)
        f.seek(0)
        header = f.read(6)
        if header != "Method":
            dw = csv.DictWriter(f, delimiter=',',
                                fieldnames=headerList)
            dw.writeheader()

        line = "{}, {}, {}, {}, {}, {}, {}, :::::::::, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n".format(
            cfg.model.type, cfg.gt.n_heads, cfg.train.batch_size,
            cfg.gt.layers, cfg.gt.dim_hidden,
            test_best_results['params'], cfg.statistics.memory,
            test_best_results['accuracy'], test_best_results['accuracy_std'],
            cfg.statistics.total_time, cfg.statistics.total_time_std,
            cfg.statistics.avg_time, cfg.statistics.avg_time_std,
            cfg.statistics.test_time, cfg.statistics.test_time_std
        )
        f.write(line)
    print("=" * 20)
    print("Writing Results File Done !!!")
