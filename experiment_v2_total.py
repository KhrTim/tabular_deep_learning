import numpy as np
from tqdm import tqdm
from mlflow.exceptions import MlflowException
from TALENT.model.utils import (
    get_deep_args,
    get_method,
    set_seeds,
)
from TALENT.model.lib.data import get_dataset
import yaml
import mlflow
import time
import os


def fold_run_exists(experiment_name, fold):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return False

    run_name = f"fold_{fold}"
    # Search for nested runs with this run name
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f'tags.mlflow.runName = "{run_name}" and attributes.status = "FINISHED"',
        output_format="list"
    )
    return len(runs) > 0


if __name__ == "__main__":
    


    indices_models = {
        "t2gformer", "realmlp", "grande", "amformer", "tabm", "trompt", "bishop",'PFN-v2'
    }
    tabr_ohe_models = {"mlp_plr", "tabr", "modernNCA"}
    remove_norm = {'PFN-v2'}

    with open("total.yaml", "r") as file:
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
                experiment_name = f"{clean_dataset}-{clean_model_name}-{max_epoch}-{cross_val_count}-NEW"
                mlflow.set_experiment(experiment_name)
                metrics_across_folds = None  # Will initialize after first predict

                for fold in tqdm(range(100, 100 + cross_val_count), desc=f"{dataset}-{model_name} folds"):
                    if fold_run_exists( experiment_name, fold):
                        print(f"FOLD {fold} IN {experiment_name} EXPERIMENT FOUND. SKIPPING...")
                        continue
                    with mlflow.start_run(run_name=f"fold_{fold}") as child_run:
                        args_list = [
                            "--dataset", dataset,
                            "--dataset_path", "data",
                            "--max_epoch", str(max_epoch),
                            "--model_type", model_name,
                            "--batch_size", str(batch_size),
                            "--gpu", str(experimental_setup['gpu']),
                            "--cross_val_fold", str(fold),
                        ]

                        if model_name in indices_models:
                            args_list += ["--cat_policy", "indices"]
                        elif model_name in tabr_ohe_models:
                            args_list += ["--cat_policy", "tabr_ohe"]
                        if model_name in remove_norm:
                            args_list += ["--normalization", "none"]

                        args, default_para, opt_space = get_deep_args(args_list)

                        
                    
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
                        if train_val_data[0]:
                            num_fea = train_val_data[0]['train'].shape[0]
                        elif train_val_data[1]:
                            num_fea = train_val_data[1]['train'].shape[0]


                        mlflow.log_param("num_features", num_fea)
                        vl, vres, metric_name, predict_logits = method.predict(test_data, info, model_name=args.evaluate_option)
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
                        mlflow.log_metric("total_time", elapsed)
