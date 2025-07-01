from tqdm import tqdm
from TALENT.model.utils import get_deep_args,show_results,tune_hyper_parameters,get_method,set_seeds
from TALENT.model.lib.data import get_dataset
import mlflow

if __name__ == '__main__':
    mlflow.pytorch.autolog()
    loss_list, results_list, time_list = [], [], []
    args,default_para,opt_space = get_deep_args()
    train_val_data,test_data,info = get_dataset(args.dataset,args.dataset_path)
    if args.tune:
        args = tune_hyper_parameters(args,opt_space,train_val_data,info)
    for seed in tqdm(range(args.seed_num)):
        args.seed = seed    # update seed  
        set_seeds(args.seed)
        method = get_method(args.model_type)(args, info['task_type'] == 'regression')
        time_cost = method.fit(train_val_data, info)    
        vl, vres, metric_name, predict_logits = method.predict(test_data, info, model_name=args.evaluate_option)
        for i, v in enumerate(metric_name):
            mlflow.log_metric(v, vres[i])

        loss_list.append(vl)
        results_list.append(vres)
        time_list.append(time_cost)

    show_results(args,info, metric_name,loss_list,results_list,time_list)
