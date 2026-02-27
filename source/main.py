import os

os.environ["WANDB_SILENT"] = "true"
os.environ["NUMEXPR_MAX_THREADS"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["RAY_AIR_NEW_OUTPUT"] = "1"

import torch
import argparse
from os.path import join
from utils.osHandler import loadyaml
from train.Trainer import Trainer, GesnTrainer
from explain.generator import ParallelGenerator
from data.dataHandler import initialize_datasets



def arg_parse():
    parser = argparse.ArgumentParser(prog='Expl',
                                     description='This application let you train and explain different types of Deep Graph Networks on different datasets')
    
    subparsers = parser.add_subparsers(help='sub-command help')
    parser_a = subparsers.add_parser('trainGraphConv', help='Train Convolutional or GESN models in parallel')
    parser_a.add_argument('--name', type=str, dest="exp_name", required=True, help="Experiment name")
    parser_a.add_argument('--exp_config', type=str, dest="exp_config", default='GINhypers.yaml', help="File containing the hyperparameters for the gridsearch")
    parser_a.add_argument('--cuda', action="store_const", const=True, default=False, help="Enable training on GPU")
    parser_a.add_argument('--GESN', action="store_const", const=True, default=False, help="Use GesnTrainer")
    parser_a.add_argument('--cpu', type=float, dest="cpu", default=2, help="Number of workers for each concurrent processes")
    parser_a.add_argument('--gpu', type=float, dest="gpu", default=0.2, help="Splits of the gpu resources across concurrent processes")
    parser_a.add_argument('--max_cpu', type=float, dest="max_cpus", default=10, help="Max number of CPUs")
    parser_a.add_argument('--max_gpu', type=float, dest="max_gpus", default=1, help="Max number of GPUs")
    parser_a.add_argument('--max_c', type=int, dest="max_concurrency", default=10, help="Maximum number of concurrent processes")
    parser_a.set_defaults(command='trainGraphConv')

    parser_b = subparsers.add_parser('explainModels', help='Explain multiple graphs in parallel')
    parser_b.add_argument('--exp_config', type=str, dest="exp_config", default='explain.yaml', help="File specifying models, explainers, metrics and discretizer")
    parser_b.add_argument('--cuda', action="store_const", const=True, default=False, help="Enable GPU usage")
    parser_b.add_argument('--max_cpu', type=float, dest="max_cpus", default=10, help="Max number of CPUs")
    parser_b.add_argument('--max_gpu', type=float, dest="max_gpus", default=2, help="Max number of GPUs")
    parser_b.set_defaults(command='explainModels')

    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()

    if args.command == 'trainGraphConv':
        exp_config = loadyaml(args.exp_config)
        cpu = args.cpu
        gpu = args.gpu
        num_cpus = args.max_cpus
        num_gpus = args.max_gpus
        max_concurrent = args.max_concurrency

        device = torch.device('cpu')
        if args.cuda:
            device = torch.device('cuda')

        exp_name = f'{args.exp_name}_{exp_config["task_properties"]["validation"]}_{exp_config["task_properties"]["kfold"]}'
        result_path = join(os.path.dirname(os.getcwd()), 'experiments', exp_name)

        initialize_datasets(dataset_configs=exp_config['datasets'],
                            ratio=exp_config['task_properties']['ratio'],
                            validation=exp_config['task_properties']['validation'], 
                            kfold=exp_config['task_properties']['kfold'],
                            stratify=exp_config['task_properties']['stratify'])

        if args.GESN:
            trainer = GesnTrainer(exp_name, exp_config, result_path, device)
        else:
            trainer = Trainer(exp_name, exp_config, result_path, device)
        results = trainer.parallel_fit(max_concurrent, num_cpus, num_gpus, cpu, gpu)

    elif args.command == 'explainModels':
        config = loadyaml(args.exp_config)
        num_cpus = args.max_cpus
        num_gpus = args.max_gpus
        device = torch.device('cuda') if args.cuda else torch.device('cpu')
        generator = ParallelGenerator(config, device, num_cpus, num_gpus)
        explanations = generator.run()
    else:

        raise Exception('Specified wrong mode')
