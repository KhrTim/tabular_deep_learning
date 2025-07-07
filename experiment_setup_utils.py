import mlflow

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

def form_args_list(dataset, model_name, visible, fold, experimental_setup, gpu_ids):
    indices_models = {"t2gformer", "realmlp", "grande", "amformer", "tabm", "trompt", "bishop", 'PFN-v2'}
    tabr_ohe_models = {"mlp_plr", "tabr", "modernNCA"}
    remove_norm = {'PFN-v2'}
    batch_size = dataset.get('batch_size', experimental_setup["batch_size"])
    args_list = [
                    "--dataset", dataset['name'],
                    "--dataset_path", "data",
                    "--max_epoch", str(experimental_setup["max_epoch"]),
                    "--model_type", model_name,
                    "--batch_size", str(batch_size),
                    "--gpu", visible,
                    "--cross_val_fold", str(fold),
                ]

    if len(gpu_ids) > 1:
        args_list += ["--multiple_gpu", visible]

    if model_name in indices_models:
        args_list += ["--cat_policy", "indices"]
    elif model_name in tabr_ohe_models:
        args_list += ["--cat_policy", "tabr_ohe"]
    if model_name in remove_norm:
        args_list += ["--normalization", "none"]

    return args_list

def create_experiment_name(dataset, model_name, experimental_setup):
    clean_dataset = dataset['name'].replace("/", "_").replace("\\", "_")
    clean_model_name = model_name.replace("/", "_").replace("\\", "_")
    experiment_name = f"{clean_dataset}-{clean_model_name}-{experimental_setup['max_epoch']}-{experimental_setup['cross_val_count']}-NEW"
    return experiment_name

def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

