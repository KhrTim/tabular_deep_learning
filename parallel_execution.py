import os
import time
import yaml
import mlflow
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from mlflow.exceptions import MlflowException
from TALENT.model.utils import get_deep_args, get_method, set_seeds
from TALENT.model.lib.data import get_dataset
import sys


def fold_run_exists(experiment_name, fold):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return False
    run_name = f"fold_{fold}"
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f'tags.mlflow.runName = "{run_name}" and attributes.status = "FINISHED"',
        output_format="list"
    )
    return len(runs) > 0

def process_datasets_on_gpu_no_logging(gpu_id, datasets, experimental_setup):
    print("!!!!!!!!!!!!!!!!!!!!!!!!! WARNING: LOG DISABLED RUN !!!!!!!!!!!!!!!!!!!!!!!!!")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[GPU {gpu_id}] Handling datasets: {datasets}")

    indices_models = {"t2gformer", "realmlp", "grande", "amformer", "tabm", "trompt", "bishop", 'PFN-v2'}
    tabr_ohe_models = {"mlp_plr", "tabr", "modernNCA"}
    remove_norm = {'PFN-v2'}

    for dataset in datasets:
        for algorithm in experimental_setup["models"]:
            model_name = algorithm["name"]
            batch_size = dataset.get('batch_size', experimental_setup["batch_size"] )

            for fold in tqdm(range(100, 100 + experimental_setup["cross_val_count"]), desc=f"{dataset['name']}-{model_name} folds"):
                args_list = [
                    "--dataset", dataset['name'],
                    "--dataset_path", "data",
                    "--max_epoch", str(experimental_setup["max_epoch"]),
                    "--model_type", model_name,
                    "--batch_size", str(batch_size),
                    "--gpu", str(gpu_id),
                    "--cross_val_fold", str(fold),
                ]

                if model_name in indices_models:
                    args_list += ["--cat_policy", "indices"]
                elif model_name in tabr_ohe_models:
                    args_list += ["--cat_policy", "tabr_ohe"]
                if model_name in remove_norm:
                    args_list += ["--normalization", "none"]

                args, _, _ = get_deep_args(args_list)


                set_seeds(experimental_setup["seed"])
                train_val_data, test_data, info = get_dataset(args.dataset, args.dataset_path, fold)
                method = get_method(args.model_type)(args, info["task_type"] == "regression")
                time_cost = method.fit(train_val_data, info)
                vl, vres, metric_name, _ = method.predict(test_data, info, model_name=args.evaluate_option)

def process_datasets_on_gpu(gpu_id, datasets, experimental_setup):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[GPU {gpu_id}] Handling datasets: {datasets}")

    indices_models = {"t2gformer", "realmlp", "grande", "amformer", "tabm", "trompt", "bishop", 'PFN-v2'}
    tabr_ohe_models = {"mlp_plr", "tabr", "modernNCA"}
    remove_norm = {'PFN-v2'}

    for dataset in datasets:
        for algorithm in experimental_setup["models"]:
            model_name = algorithm["name"]
            batch_size = dataset.get('batch_size', experimental_setup["batch_size"] )
            clean_dataset = dataset['name'].replace("/", "_").replace("\\", "_")
            clean_model_name = model_name.replace("/", "_").replace("\\", "_")
            experiment_name = f"{clean_dataset}-{clean_model_name}-{experimental_setup['max_epoch']}-{experimental_setup['cross_val_count']}-NEW"
            mlflow.set_experiment(experiment_name)
            metrics_across_folds = {}

            for fold in tqdm(range(100, 100 + experimental_setup["cross_val_count"]), desc=f"{dataset['name']}-{model_name} folds"):
                if fold_run_exists(experiment_name, fold):
                    print(f"[GPU {gpu_id}] FOLD {fold} IN {experiment_name} FOUND. SKIPPING...")
                    continue

                with mlflow.start_run(run_name=f"fold_{fold}") as child_run:
                    args_list = [
                        "--dataset", dataset['name'],
                        "--dataset_path", "data",
                        "--max_epoch", str(experimental_setup["max_epoch"]),
                        "--model_type", model_name,
                        "--batch_size", str(batch_size),
                        "--gpu", str(gpu_id),
                        "--cross_val_fold", str(fold),
                    ]

                    if model_name in indices_models:
                        args_list += ["--cat_policy", "indices"]
                    elif model_name in tabr_ohe_models:
                        args_list += ["--cat_policy", "tabr_ohe"]
                    if model_name in remove_norm:
                        args_list += ["--normalization", "none"]

                    args, _, _ = get_deep_args(args_list)

                    for k, v in vars(args).items():
                        try:
                            mlflow.log_param(k, v)
                        except MlflowException:
                            pass

                    mlflow.log_param("fold", fold)
                    mlflow.log_param("model", model_name)

                    set_seeds(experimental_setup["seed"])
                    train_val_data, test_data, info = get_dataset(args.dataset, args.dataset_path, fold)
                    method = get_method(args.model_type)(args, info["task_type"] == "regression")
                    time_cost = method.fit(train_val_data, info)
                    num_fea = train_val_data[0]['train'].shape[1] if train_val_data[0] else train_val_data[1]['train'].shape[1]
                    mlflow.log_param("num_features", num_fea)

                    start_time = time.time()
                    vl, vres, metric_name, _ = method.predict(test_data, info, model_name=args.evaluate_option)
                    elapsed = time.time() - start_time

                    if metrics_across_folds is None:
                        metrics_across_folds = {name: [] for name in metric_name}
                        metrics_across_folds['time'] = []

                    for name, val in zip(metric_name, vres):
                        metrics_across_folds[name].append(val)
                    metrics_across_folds['time'].append(time_cost)

                    mlflow.log_metric("val_loss", vl)
                    for name, val in zip(metric_name, vres):
                        mlflow.log_metric(name, val)
                    mlflow.log_metric("train_time", time_cost)
                    mlflow.log_metric("inference_time", elapsed)



def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]

    with open(config_path, "r") as file:
        experimental_setup = yaml.safe_load(file)

    log_disabled_run = True
    all_datasets = experimental_setup["datasets"]
    gpu_list = experimental_setup.get("gpus", [0])
    dataset_chunks = split_list(all_datasets, len(gpu_list))
    processes = []
    for gpu_id, dataset_subset in zip(gpu_list, dataset_chunks):
        p = mp.Process(target=process_datasets_on_gpu if not log_disabled_run else process_datasets_on_gpu_no_logging, args=(gpu_id, dataset_subset, experimental_setup))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
