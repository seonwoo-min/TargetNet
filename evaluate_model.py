# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

import os
import sys
import argparse
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import src.config as config
from src.data import get_dataset_from_configs
from src.model.model_utils import get_model
from src.train import Trainer
from src.utils import Print, set_seeds, set_output, check_args

parser = argparse.ArgumentParser('Evaluate a microRNA Target Prediction Model')
parser.add_argument('--data-config',  help='path for data configuration file')
parser.add_argument('--model-config', help='path for model configuration file')
parser.add_argument('--run-config', help='path for run configuration file')
parser.add_argument('--checkpoint', help='path for checkpoint to resume')
parser.add_argument('--output-path', help='path for outputs (default: stdout and without saving)')


def main():
    args = vars(parser.parse_args())
    check_args(args)
    set_seeds(2020)
    data_cfg = config.DataConfig(args["data_config"])
    model_cfg = config.ModelConfig(args["model_config"])
    run_cfg   = config.RunConfig(args["run_config"], eval=True)
    output, save_prefix = set_output(args, "evaluate_model_log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.print_configs(args, [data_cfg, model_cfg, run_cfg], device, output)
    torch.zeros((1)).to(device)

    ## Loading datasets
    start = Print(" ".join(['start loading datasets']), output)
    dataset_idxs, datasets, iterators = data_cfg.path.keys(), [], []
    for idx in dataset_idxs:
        dataset = get_dataset_from_configs(data_cfg, idx)
        iterator = torch.utils.data.DataLoader(dataset, run_cfg.batch_size, shuffle=False, pin_memory=True, num_workers=4)
        datasets.append(dataset)
        iterators.append(iterator)
        end = Print(" ".join(['loaded', str(len(dataset)), idx, 'samples']), output)
    Print(" ".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## initialize a model
    start = Print('start initializing a model', output)
    model, params = get_model(model_cfg, data_cfg.with_esa)
    end = Print('end initializing a model', output)
    Print(" ".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## setup trainer configurations
    start = Print('start setting trainer configurations', output)
    trainer = Trainer(model)
    trainer.load_model(args["checkpoint"], output)
    trainer.set_device(device)
    end = Print('end setting trainer configurations', output)
    Print(" ".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## evaluate a model
    start = Print('start evaluating a model', output)
    ### validation
    for idx, dataset, iterator in zip(dataset_idxs, datasets, iterators):
        Print(" ".join(['processing', idx]), output)

        ### validation
        for B, batch in enumerate(iterator):
            trainer.evaluate(batch, device)
            if B % 5 == 0: print('# {} {:.1%}'.format(idx, B / len(iterator)), end='\r', file=sys.stderr)
        print(' ' * 150, end='\r', file=sys.stderr)

        ### save outputs
        trainer.aggregate(dataset.set_labels)
        trainer.save_outputs(idx, save_prefix)

    end = Print('end evaluating a model', output)
    Print(" ".join(['elapsed time:', str(end - start)]), output, newline=True)
    if not output == sys.stdout: output.close()

if __name__ == '__main__':
    main()
