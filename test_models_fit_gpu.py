import numpy as np
from tqdm import tqdm
from TALENT.model.utils import (
    get_deep_args,
    get_method,
    set_seeds,
)
from TALENT.model.lib.data import get_dataset
import yaml
import time
import os
import torch

if __name__ == "__main__":
    indices_models = {
        "t2gformer", "realmlp", "grande", "amformer", "tabm", "trompt", "bishop",'PFN-v2'
    }
    tabr_ohe_models = {"mlp_plr", "tabr", "modernNCA"}
    remove_norm = {'PFN-v2'}
    gpu_num = 1
    with open("experiment_setup.yaml", "r") as file:
        experimental_setup = yaml.safe_load(file)

    default_batch_size = experimental_setup["batch_size"]
    seed = experimental_setup["seed"]
    max_epoch = 1
    cross_val_count = 1


    for dataset in experimental_setup["datasets"]:
        for algorithm in experimental_setup["models"]:
                model_name = algorithm["name"]
                batch_size = dataset.get('batch_size', experimental_setup["batch_size"])

                print(f"TESTING MODEL {model_name} ON DATASET {dataset['name']} BATCH {batch_size}")

                for fold in tqdm(range(100, 100 + cross_val_count), desc=f"{dataset['name']}-{model_name} folds"):
                    args_list = [
                        "--dataset", dataset['name'],
                        "--dataset_path", "data",
                        "--max_epoch", str(max_epoch),
                        "--model_type", model_name,
                        "--batch_size", str(batch_size),
                        "--gpu", str(gpu_num),
                        "--cross_val_fold", str(fold),
                    ]

                    if model_name in indices_models:
                        args_list += ["--cat_policy", "indices"]
                    elif model_name in tabr_ohe_models:
                        args_list += ["--cat_policy", "tabr_ohe"]
                    if model_name in remove_norm:
                        args_list += ["--normalization", "none"]

                    args, default_para, opt_space = get_deep_args(args_list)

                    set_seeds(seed)
                    train_val_data, test_data, info = get_dataset(args.dataset, args.dataset_path, fold)
                    method = get_method(args.model_type)(args, info["task_type"] == "regression")
                    time_cost = method.fit(train_val_data, info)