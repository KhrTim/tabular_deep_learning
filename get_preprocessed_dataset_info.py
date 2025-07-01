from tqdm import tqdm
from TALENT.model.utils import (
    get_deep_args,
    set_seeds,
)
from TALENT.model.lib.data import get_dataset
import yaml
import os
import pprint



if __name__ == "__main__":
    indices_models = {
        "t2gformer", "realmlp", "grande", "amformer", "tabm", "trompt", "bishop",'PFN-v2'
    }
    tabr_ohe_models = {"mlp_plr", "tabr", "modernNCA"}
    remove_norm = {'PFN-v2'}

    with open("total.yaml", "r") as file:
        experimental_setup = yaml.safe_load(file)

    default_batch_size = experimental_setup["batch_size"]
    seed = experimental_setup["seed"]
    max_epoch = experimental_setup["max_epoch"]
    cross_val_count = experimental_setup["cross_val_count"]

    res = {}
    for dataset in experimental_setup["datasets"]:
        if dataset not in res:
             res[dataset] = {}
        for algorithm in experimental_setup["models"]:
                model_name = algorithm["name"]
                batch_size = algorithm.get("params", {}).get("batch_size", default_batch_size)
                clean_dataset = dataset.replace("/", "_").replace("\\", "_")
                clean_model_name = model_name.replace("/", "_").replace("\\", "_")
                experiment_name = f"{clean_dataset}-{clean_model_name}-{max_epoch}-{cross_val_count}-NEW"

                args_list = [
                    "--dataset", dataset,
                    "--dataset_path", "data",
                    "--max_epoch", str(max_epoch),
                    "--model_type", model_name,
                    "--batch_size", str(batch_size),
                    "--gpu", str(experimental_setup['gpu']),
                ]

                if model_name in indices_models:
                    args_list += ["--cat_policy", "indices"]
                elif model_name in tabr_ohe_models:
                    args_list += ["--cat_policy", "tabr_ohe"]
                if model_name in remove_norm:
                    args_list += ["--normalization", "none"]

                args, default_para, opt_space = get_deep_args(args_list)
                set_seeds(seed)
                train_val_data, test_data, info = get_dataset(args.dataset, args.dataset_path)
                res[dataset][model_name] = train_val_data[0]['train'].shape
    pprint.pp(res)

                        
