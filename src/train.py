# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

import os
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F

from src.utils import Print
from src.evaluate import compute_metrics


class Trainer():
    """ train / eval helper class """
    def __init__(self, model):
        self.model = model
        self.optim = None
        self.scheduler = None

        # initialize logging parameters
        self.train_flag = False
        self.epoch = 0.0
        self.best_loss = None
        self.logger_train = Logger()
        self.logger_eval  = Logger()

    def train(self, batch, device):
        # training of the model
        batch = set_device(batch, device)

        self.model.train()
        self.optim.zero_grad()
        inputs1, inputs2, labels = batch
        outputs = self.model(inputs1, inputs2)
        loss = get_loss(outputs, labels)
        loss.backward()
        self.optim.step()

        # logging
        outputs = torch.sigmoid(outputs)
        self.logger_train.update(len(outputs), loss.item())
        self.logger_train.keep(outputs, labels)

    def evaluate(self, batch, device):
        # evaluation of the model
        batch = set_device(batch, device)

        self.model.eval()
        with torch.no_grad():
            inputs1, inputs2, labels = batch
            outputs = self.model(inputs1, inputs2)
            loss = get_loss(outputs, labels)

        # logging
        outputs = torch.sigmoid(outputs)
        self.logger_eval.update(len(outputs), loss.item())
        self.logger_eval.keep(outputs, labels)

    def scheduler_step(self):
        # scheduler_step
        self.scheduler.step(self.logger_eval.get_loss())

    def save_model(self, save_prefix):
        # save state_dicts to checkpoint """
        if save_prefix is None: return
        elif not os.path.exists(save_prefix + "/checkpoints/"):
            os.makedirs(save_prefix + "/checkpoints/", exist_ok=True)

        state = {}
        state["model"] = self.model.state_dict()
        state["optim"] = self.optim.state_dict()
        state["scheduler"] = self.scheduler.state_dict()
        state["epoch"] = self.epoch
        torch.save(state, save_prefix + "/checkpoints/%d.pt" % self.epoch)

        loss = self.logger_eval.get_loss()
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            torch.save(state, save_prefix + "/checkpoints/best_loss.pt")

    def load_model(self, checkpoint, save_prefix, output):
        # load state_dicts from checkpoint """
        if checkpoint is None:
            if save_prefix is None or not os.path.exists(save_prefix + "/checkpoints/"): return
            checkpoints = [os.path.splitext(file)[0] for file in os.listdir(save_prefix + "/checkpoints/")]
            checkpoints = sorted([int(checkpoint) for checkpoint in checkpoints if not checkpoint.startswith("best")])
            if len(checkpoints) == 0: return
            checkpoint = save_prefix + "/checkpoints/%d.pt" % checkpoints[-1]
            Print('resuming from the last checkpoint [%s]' % (checkpoint), output)

        Print('loading a model state_dict from the checkpoint', output)
        checkpoint = torch.load(checkpoint, map_location="cpu")
        state_dict = OrderedDict()
        for k, v in checkpoint["model"].items():
            if k.startswith("module."): k = k[7:]
            state_dict[k] = v
        self.model.load_state_dict(state_dict)
        if self.optim is not None and "optim" in checkpoint:
            Print('loading a optim state_dict from the checkpoint', output)
            self.optim.load_state_dict(checkpoint["optim"])
            Print('loading a scheduler state_dict from the checkpoint', output)
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            Print('loading current epoch from the checkpoint', output)
            self.epoch = checkpoint["epoch"]

    def save_outputs(self, idx, save_prefix):
        # save validation output
        if not os.path.exists(save_prefix + "/outputs/"):
            os.makedirs(save_prefix + "/outputs/", exist_ok=True)
        self.logger_eval.aggregate()
        np.save(save_prefix + "/outputs/%s_outputs.npy" % (idx), self.logger_eval.outputs)
        np.save(save_prefix + "/outputs/%s_labels.npy" % (idx), self.logger_eval.labels)

    def set_device(self, device):
        # set gpu configurations
        self.model = self.model.to(device)

    def set_optim_scheduler(self, run_cfg, params):
        # set optim and scheduler for training
        optim, scheduler = get_optim_scheduler(run_cfg, params)
        self.train_flag = True
        self.optim = optim
        self.scheduler = scheduler

    def get_headline(self):
        # get a headline for logging
        headline = []
        if self.train_flag:
            headline += ["ep", "split"]
            headline += self.logger_train.get_headline(loss_only=True)
            headline += ["|"]

        headline += ["split"]
        headline += self.logger_eval.get_headline(loss_only=not self.train_flag)

        return "\t".join(headline)

    def log(self, idx, output, writer):
        # logging
        log, log_dict = [], {}

        if self.train_flag:
            self.logger_train.evaluate(loss_only=True)
            log += ["%03d" % self.epoch, "train"]
            log += self.logger_train.log
            if writer is not None:
                for k, v in self.logger_train.log_dict.items():
                    if k not in log_dict: log_dict[k] = {}
                    log_dict[k]["train"] = v
            log += ["|"]

        self.logger_eval.evaluate(loss_only=not self.train_flag)
        log += [idx]
        log += self.logger_eval.log
        if writer is not None:
            for k, v in self.logger_eval.log_dict.items():
                if k not in log_dict: log_dict[k] = {}
                log_dict[k][idx] = v

        Print("\t".join(log), output)
        if writer is not None:
            for k, v in log_dict.items():
                writer.add_scalars(k, v, self.epoch)
            writer.flush()

        self.log_reset()

    def log_reset(self):
        # reset logging parameters
        self.logger_train.reset()
        self.logger_eval.reset()


class Logger():
    """ Logger class """
    def __init__(self):
        self.total = 0.0
        self.loss = 0.0
        self.outputs = []
        self.labels = []
        self.log = []
        self.log_dict = {}

    def update(self, total, loss):
        # update logger for current mini-batch
        self.total += total
        self.loss += loss * total

    def keep(self, outputs, labels):
        # keep labels and outputs for future computations
        self.outputs.append(outputs.cpu().detach().numpy())
        self.labels.append(labels.cpu().detach().numpy())

    def get_loss(self):
        # get current averaged loss
        loss = self.loss / self.total
        return loss

    def get_headline(self, loss_only):
        # get headline
        headline = ["loss"]
        if not loss_only: headline += ["tp", "fp", "fn", "tn", "acc", "pr", "re", "np", "sp", "f1"]

        return headline

    def evaluate(self, loss_only=False):
        # compute evaluation metrics
        self.aggregate()
        metrics = ["loss"]
        evaluations = [self.get_loss()]

        if not loss_only:
            metrics += ["tp", "fp", "fn", "tn", "acc", "pr", "re", "np", "sp", "f1"]
            evaluations += [*compute_metrics(self.labels, self.outputs > 0.5)]
        self.log = ["%.4f" % eval if e == 0 or e > 4 else "%d" % eval for e, eval in enumerate(evaluations)]
        self.log_dict = {metric:eval for metric, eval in zip(metrics, evaluations)}

    def aggregate(self):
        # aggregate kept labels and outputs
        if isinstance(self.outputs, list) and len(self.outputs) > 0:
            self.outputs = np.concatenate(self.outputs, axis=0)
        if isinstance(self.labels, list) and len(self.labels) > 0:
            self.labels = np.concatenate(self.labels, axis=0)

    def reset(self):
        # reset logger
        self.total = 0.0
        self.loss = 0.0
        self.outputs = []
        self.labels = []
        self.log = []
        self.log_dict = {}


def get_optim_scheduler(cfg, params):
    """ configure optim and scheduler """
    optim = torch.optim.Adam([{'params': params[0], 'weight_decay': cfg.weight_decay},
                              {'params': params[1], 'weight_decay': 0}], lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, "min", 0.2, 5)

    return optim, scheduler


def get_loss(outputs, labels):
    """ get (binary) cross entropy loss """
    loss = -torch.mean(labels * F.logsigmoid(outputs) + (1 - labels) * F.logsigmoid(-outputs))

    return loss


def set_device(batch, device):
    """ recursive function for setting device for batch """
    if isinstance(batch, tuple) or isinstance(batch, list):
        return [set_device(t, device) for t in batch]
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        return batch
