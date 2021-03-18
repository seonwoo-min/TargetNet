# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

import faulthandler
faulthandler.enable()

import os
import sys
import argparse
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import torch

import src.config as config
from src.data import get_dataset_from_configs
from src.model.model_utils import get_model
from src.train import Trainer
from src.utils import Print, set_seeds, set_output, check_args

parser = argparse.ArgumentParser('Train a microRNA Target Prediction Model')
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
    run_cfg   = config.RunConfig(args["run_config"], eval=False)
    output, save_prefix = set_output(args, "train_model_log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.print_configs(args, [data_cfg, model_cfg, run_cfg], device, output)
    torch.zeros((1)).to(device)

    ## Loading datasets
    start = Print(" ".join(['start loading datasets']), output)
    dataset_train = get_dataset_from_configs(data_cfg, split_idx="train")
    dataset_val   = get_dataset_from_configs(data_cfg, split_idx="val")
    iterator_train = torch.utils.data.DataLoader(dataset_train, run_cfg.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    iterator_val   = torch.utils.data.DataLoader(dataset_val,   run_cfg.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    end = Print(" ".join(['loaded', str(len(dataset_train)), 'dataset_train samples']), output)
    end = Print(" ".join(['loaded', str(len(dataset_val )),  'dataset_val samples']), output)
    Print(" ".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## initialize a model
    start = Print('start initializing a model', output)
    model, params = get_model(model_cfg, data_cfg.with_esa, run_cfg.dropout_rate)
    end = Print('end initializing a model', output)
    Print(" ".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## setup trainer configurations
    start = Print('start setting trainer configurations', output)
    trainer = Trainer(model)
    trainer.set_device(device)
    trainer.set_optim_scheduler(run_cfg, params)
    end = Print('end setting trainer configurations', output)
    Print(" ".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## train a model
    start = Print('start training a model', output)
    Print(trainer.get_headline(), output)
    for epoch in range(int(trainer.epoch), run_cfg.num_epochs):
        ### train
        for B, batch in enumerate(iterator_train):
            trainer.train(batch, device)
            if B % 5 == 0: print('# epoch [{}/{}] train {:.1%}'.format(
                epoch + 1, run_cfg.num_epochs, B / len(iterator_train)), end='\r', file=sys.stderr)
        print(' ' * 150, end='\r', file=sys.stderr)

        ### validation
        for B, batch in enumerate(iterator_val):
            trainer.evaluate(batch, device)
            if B % 5 == 0: print('# epoch [{}/{}] val {:.1%}'.format(
                epoch + 1, run_cfg.num_epochs, B / len(iterator_val)), end='\r', file=sys.stderr)
        print(' ' * 150, end='\r', file=sys.stderr)

        ### print log and save models
        trainer.epoch += 1
        trainer.scheduler_step()
        trainer.save_model(save_prefix)
        trainer.log("val", output)

    end = Print('end training a model', output)
    Print(" ".join(['elapsed time:', str(end - start)]), output, newline=True)
    if not output == sys.stdout: output.close()

if __name__ == '__main__':
    main()
