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
from contextlib import nullcontext
from experiment_setup_utils import create_experiment_name, fold_run_exists, form_args_list, split_list




def process_model_on_gpus(gpu_ids, model, experimental_setup):
    print(model)
    save_results_to_csv = experimental_setup.get("save_results_to_csv", False)
    log_to_mlflow = experimental_setup.get("log_to_mlflow", True)
    only_evaluation = experimental_setup.get("eval_only", False)
    experimental_seed = experimental_setup.get("seed", 1)
    if isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]
    
    visible = ",".join(map(str, gpu_ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = visible

    print(f"[GPUs {visible}] Running model: {model['name']}")

    if not log_to_mlflow:
        print("!!!!!!!!!!!!!!!!!!!!!!!!! WARNING: MLFLOW LOG DISABLED RUN !!!!!!!!!!!!!!!!!!!!!!!!!")
    if save_results_to_csv:
        print("!!!!!!!!!!!!!!!!!!!!!!! RESULTS WILL BE SAVED TO THE CSV FILE !!!!!!!!!!!!!!!!!!!!!!!")
    if only_evaluation:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! EVAL ONLY MODE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    model_name = model["name"]
    for dataset in experimental_setup["datasets"]:
        experiment_name = create_experiment_name(dataset, model_name, experimental_setup)
        
        if log_to_mlflow:
            mlflow.set_experiment(experiment_name)
        
        metrics_across_folds = None

        for fold in tqdm(range(100, 100 + experimental_setup["cross_val_count"]), desc=f"{dataset['name']}-{model_name} folds"):
            if fold_run_exists(experiment_name, fold) and not log_to_mlflow:
                print(f"[GPU {gpu_ids[0]}] FOLD {fold} IN {experiment_name} FOUND. SKIPPING...")
                continue

            args_list = form_args_list(dataset, model_name, visible, fold, experimental_setup, gpu_ids)
            if log_to_mlflow:
                run_context = mlflow.start_run(run_name=f"fold_{fold}")
            else:
                run_context = nullcontext()

            with run_context:
                args, _, _ = get_deep_args(args_list)

                if log_to_mlflow:
                    for k, v in vars(args).items():
                        try:
                            mlflow.log_param(k, v)
                        except MlflowException:
                            pass
                    mlflow.log_param("fold", fold)
                    mlflow.log_param("model", model_name)
                    mlflow.log_param("gpu_ids", visible)

                
                set_seeds(experimental_seed)
                print("---------> LOADING THE DATASET")
                train_val_data, test_data, info = get_dataset(args.dataset, args.dataset_path, fold)
                print("---------> LOADING THE MODEL")
                method = get_method(args.model_type)(args, info["task_type"] == "regression")
                print("---------> TRAINING THE MODEL")
                # time_cost = method.fit(train_val_data, info)
                if not only_evaluation:
                    time_cost = method.fit(train_val_data, info)
                else:
                    time_cost = method.fit(train_val_data, info,train=False)

                if not save_results_to_csv:
                    start_time = time.time()
                    vl, vres, metric_name, _ = method.predict(test_data, info, model_name=args.evaluate_option)
                    elapsed = time.time() - start_time
                else:
                    start_time = time.time()
                    vl, vres, metric_name, _ = method.predict_and_safe_to_file(test_data, info, model_name=args.evaluate_option)
                    elapsed = time.time() - start_time

                if metrics_across_folds is None:
                    metrics_across_folds = {name: [] for name in metric_name}
                    metrics_across_folds['time'] = []

                for name, val in zip(metric_name, vres):
                    metrics_across_folds[name].append(val)
                metrics_across_folds['time'].append(time_cost)

                if log_to_mlflow:
                    mlflow.log_metric("val_loss", vl)
                    for name, val in zip(metric_name, vres):
                        mlflow.log_metric(name, val)
                    mlflow.log_metric("train_time", time_cost)
                    mlflow.log_metric("inference_time", elapsed)



def process_models_on_gpu(gpu_id, models, dataset_split, experimental_setup):
    # Update setup
    current_setup = experimental_setup.copy()
    current_setup['datasets'] = dataset_split

    for model in models:
        process_model_on_gpus(gpu_id, model, current_setup)

def run_single_gpu_models(datasets_list, gpu_ids, models_list, experimental_setup):
    processes = []
    dataset_splits = split_list(datasets_list, len(gpu_ids))

    for gpu_idx, dataset_split in enumerate(dataset_splits):
        p = mp.Process(
            target=process_models_on_gpu,
            args=(gpu_ids[gpu_idx], models_list, dataset_split, experimental_setup)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    # print("Available GPUs:", torch.cuda.device_count())
    # mp.set_start_method('spawn')


    if len(sys.argv) < 2:
        print("Usage: python script.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as file:
        experimental_setup = yaml.safe_load(file)

    multi_gpu_models = {"modernNCA", "tabr","amformer", "t2gformer"}
    large_datasets = {"imdb_drama", "hcdr_main", "SantanderCustomerSatisfaction"}
    
    large_datasets_in_cofig  =  [m for m in experimental_setup['datasets'] if m.get('name') in large_datasets]
    other_datasets_in_config  = [m for m in experimental_setup['datasets'] if m.get('name') not in large_datasets]
    multi_gpu_matching_models = [m for m in experimental_setup['models'] if m.get('name') in multi_gpu_models]
    single_gpu_matching_models = [m for m in experimental_setup['models'] if m.get('name') not in multi_gpu_models]

    print(experimental_setup)
    print('-----------')
    print(large_datasets_in_cofig)

    gpu_ids = experimental_setup['gpus']
    if not isinstance(gpu_ids, list):
        gpu_ids = list(gpu_ids)

    if multi_gpu_matching_models and large_datasets_in_cofig:
        large_model_large_dataset_cofig = experimental_setup.copy()
        large_model_large_dataset_cofig['datasets'] = large_datasets_in_cofig
        print(experimental_setup)
        #TODO: check why 2 gpus are not enough
        # if len(gpu_ids) > 1:
        #     for model in multi_gpu_matching_models:
        #         process_model_on_gpus(gpu_ids, model, experimental_setup)
        # else:
        #     print("Not enough gpus to use a large model on a large dataset")

    if single_gpu_matching_models and large_datasets_in_cofig:
        large_model_large_dataset_cofig = experimental_setup.copy()
        large_model_large_dataset_cofig['datasets'] = large_datasets_in_cofig
        run_single_gpu_models(large_datasets_in_cofig, gpu_ids, single_gpu_matching_models, large_model_large_dataset_cofig)
    
    experimental_setup['datasets'] = other_datasets_in_config
    run_single_gpu_models(other_datasets_in_config, gpu_ids, experimental_setup['models'], experimental_setup)