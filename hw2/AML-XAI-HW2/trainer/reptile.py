import copy
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import trainer


def _calculate_weights_diff(new_net, old_net):
    old_net_weights = {n: p for n, p in old_net.named_parameters()}
    diffs = {name: p - old_net_weights[name] for name, p in new_net.named_parameters()}
    return diffs


def _add_named_parameters(named_param1, named_param2):
    for name, p in named_param2.items():
        named_param1[name] += p

    return named_param1


def _update_network_parameters(old_net, weights_diff, inner_step, lr):
    updated_weights = {}
    for name, p in old_net.named_parameters():
        updated_weights[name] = p + lr * (weights_diff[name] / inner_step)

    old_net.load_state_dict(updated_weights)


def _to_numpy(ts):
    out = []
    for t in ts:
        if isinstance(t, torch.Tensor):
            out.append(t.cpu())
        else:
            out.append(t)
    assert len(out) == len(ts)
    return np.array(out)


class Trainer(trainer.GenericTrainer):
    """
    Meta Learner
    """

    def __init__(self, model, args):
        """

        :param args:
        """
        super(Trainer, self).__init__(model, args)

        if self.args.dataset == 'omniglot':
            self.loss = nn.CrossEntropyLoss()
        elif self.args.dataset == 'sine':
            self.loss = nn.MSELoss()

    def _train_epoch(self, x_spt, y_spt, x_qry, y_qry):
        """
        You should combine x_spt and y_spt to make data loader that produces mini-batches
        :param x_spt:   [b, setsz, c_, h, w] or [b, setsz, 1]
         - Training input data
        :param y_spt:   [b, setsz] or [b, setsz, 1]
         - Training target data
        :param x_qry:   [b, querysz, c_, h, w] or [b, setsz, 1]
         - Test input data
        :param y_qry:   [b, querysz] or [b, setsz, 1]
         - Test target data
        :return: 'results' (a list)
        """

        # results for meta-training
        # Sine wave: MSE loss for all tasks
        # Omniglot: Average accuracy for all tasks
        # In a list 'results', it should contain MSE loss or accuracy computed at each inner loop step.
        # The components in 'results' are as follows:
        # results[0]: results for pre-update model
        # results[1:]: results for the adapted model at each inner loop step
        corrects = [0 for _ in range(self.inner_step + 1)]

        x = torch.cat((x_spt, x_qry), 1)
        y = torch.cat((y_spt, y_qry), 1)

        task_num = x_spt.size(0)

        weights_diff = {n: torch.zeros_like(p) for n, p in self.net.named_parameters()}
        for t in range(task_num):
            # pre-update
            with torch.no_grad():
                loss_q, correct = self._evaluate(self.net, x_qry[t], y_qry[t])
                corrects[0] += correct

            copied_net = copy.deepcopy(self.net)
            optimizer = optim.Adam(copied_net.parameters(), lr=self.inner_lr, betas=(0, 0.999))
            for i in range(self.inner_step):
                optimizer.zero_grad()
                # the first update
                logits = copied_net(x[t])
                loss = self.loss(logits, y[t])
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    loss_q, correct = self._evaluate(copied_net, x_qry[t], y_qry[t])
                    corrects[i + 1] += correct

                curr_diff = _calculate_weights_diff(copied_net, self.net)
                weights_diff = _add_named_parameters(weights_diff, curr_diff)

        _update_network_parameters(self.net, weights_diff, self.inner_step, self.meta_lr)
        querysz = x_qry.size(1)
        accuracies = _to_numpy(corrects) / (querysz * task_num)

        return accuracies

    def _evaluate(self, network, x, y):
        logits = network(x)
        loss = self.loss(logits, y)

        preds = F.softmax(logits, dim=1).argmax(dim=1)
        correct = torch.eq(preds, y).sum()

        return loss, correct

    def _finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [setsz, c_, h, w] or [setsz, 1]
         - Training input data
        :param y_spt:   [setsz] or [setsz, 1]
         - Training target data
        :param x_qry:   [querysz, c_, h, w] or [querysz, 1]
         - Test input data
        :param y_qry:   [querysz] or [querysz, 1]
         - Test target data
        :return: 'results' (a list)
        """

        # results for meta-test
        # Sine wave: MSE loss for current task
        # Omniglot: Average accuracy for current task
        # In a list 'results', it should contain MSE loss or accuracy computed at each inner loop step.
        # The components in 'results' are as follows:
        # results[0]: results for pre-update model
        # results[1:]: results for the adapted model at each inner loop step
        corrects = [0 for _ in range(self.inner_step_test + 1)]
        losses_q = [0 for _ in range(self.inner_step_test + 1)]

        net = deepcopy(self.net)
        optimizer = optim.Adam(net.parameters(), lr=self.inner_lr, betas=(0, 0.999))

        # pre-update
        with torch.no_grad():
            loss_q, correct = self._evaluate(net, x_qry, y_qry)
            losses_q[0] += loss_q.item()
            corrects[0] += correct.item()

        for i in range(self.inner_step_test):
            # the first update
            optimizer.zero_grad()
            logits = net(x_spt)
            loss = self.loss(logits, y_spt)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                loss_q, correct = self._evaluate(net, x_qry, y_qry)
                losses_q[i + 1] += loss_q.item()
                corrects[i + 1] += correct.item()

        querysz = x_qry.size(0)
        accuracies = _to_numpy(corrects) / querysz
        losses_q = _to_numpy(losses_q)

        return accuracies, losses_q
