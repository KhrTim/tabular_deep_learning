import numpy as np
from tqdm import tqdm
from mlflow.exceptions import MlflowException
from TALENT.model.utils import (
    get_deep_args,
    show_results,
    tune_hyper_parameters,
    get_method,
    set_seeds,
)
from TALENT.model.lib.data import get_dataset
import yaml
import mlflow
import time
import csv
import os

if __name__ == "__main__":

    backup_dir = "cv_results"
    os.makedirs(backup_dir, exist_ok=True)
    


    indices_models = {
        "t2gformer", "realmlp", "grande", "amformer", "tabm", "trompt", "bishop",'PFN-v2'
    }
    tabr_ohe_models = {"mlp_plr", "tabr", "modernNCA"}
    remove_norm = {'PFN-v2'}

    with open("experiment_setup.yaml", "r") as file:
        experimental_setup = yaml.safe_load(file)
    print(experimental_setup)

    default_batch_size = experimental_setup["batch_size"]
    seed = experimental_setup["seed"]
    max_epoch = experimental_setup["max_epoch"]
    cross_val_count = experimental_setup["cross_val_count"]

    for dataset in experimental_setup["datasets"]:
        for algorithm in experimental_setup["models"]:
            model_name = algorithm["name"]
            batch_size = algorithm.get("params", {}).get("batch_size", default_batch_size)
            clean_dataset = dataset.replace("/", "_").replace("\\", "_")
            clean_model_name = model_name.replace("/", "_").replace("\\", "_")
            csv_path = os.path.join(backup_dir, f"{dataset}_{model_name}_cv_metrics.csv")
            mlflow.set_experiment(f"{clean_dataset}-{clean_model_name}-{max_epoch}-{cross_val_count}")

            metrics_across_folds = None  # Will initialize after first predict

            for fold in tqdm(range(100, 100 + cross_val_count), desc=f"{dataset}-{model_name} folds"):
                args_list = [
                    "--dataset", dataset,
                    "--dataset_path", "data",
                    "--max_epoch", str(max_epoch),
                    "--model_type", model_name,
                    "--batch_size", str(batch_size),
                    "--gpu", "0",
                    "--cross_val_fold", str(fold),
                ]

                if model_name in indices_models:
                    args_list += ["--cat_policy", "indices"]
                elif model_name in tabr_ohe_models:
                    args_list += ["--cat_policy", "tabr_ohe"]
                if model_name in remove_norm:
                    args_list += ["--normalization", "none"]

                args, default_para, opt_space = get_deep_args(args_list)

                
                run_name = f"{dataset}-{model_name}-fold{fold}"
                with mlflow.start_run(run_name=run_name):

                    # Log params safely
                    for k, v in vars(args).items():
                        try:
                            mlflow.log_param(k, v)
                        except MlflowException:
                            pass

                    mlflow.log_param("fold", fold)
                    mlflow.log_param("model", model_name)

                    set_seeds(seed)
                    train_val_data, test_data, info = get_dataset(args.dataset, args.dataset_path, fold)
                    method = get_method(args.model_type)(args, info["task_type"] == "regression")

                    start_time = time.time()
                    time_cost = method.fit(train_val_data, info)
                    vl, vres, metric_name, predict_logits = method.predict(test_data, info, model_name=args.evaluate_option)
                    elapsed = time.time() - start_time

                    if metrics_across_folds is None:
                        metrics_across_folds = {name: [] for name in metric_name}

                    for name, val in zip(metric_name, vres):
                        metrics_across_folds[name].append(val)

                    mlflow.log_metric("val_loss", vl)
                    for name, val in zip(metric_name, vres):
                        mlflow.log_metric(name, val)
                    mlflow.log_metric("train_time", time_cost)
                    mlflow.log_metric("total_time", elapsed)

                    print(f"[✔] {run_name} completed.")
            with open(csv_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Metric", "Fold_Values", "Mean", "Std"])
                for name, values in metrics_across_folds.items():
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    writer.writerow([name, values, mean_val, std_val])

            # Log aggregate metrics in a separate MLflow run
            with mlflow.start_run(run_name=f"{dataset}-{model_name}-aggregate", nested=True):
                for name, values in metrics_across_folds.items():
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    print(f"{name}: Mean = {mean_val:.4f}, Std = {std_val:.4f}")
                    mlflow.log_metric(f"{name}_mean", mean_val)
                    mlflow.log_metric(f"{name}_std", std_val)