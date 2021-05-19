from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import trainer


class Trainer(trainer.GenericTrainer):
    """
    Meta Learner
    """

    def __init__(self, model, args):
        """
        :param args:
        """
        super(Trainer, self).__init__(model, args)

        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.inner_optim = optim.SGD(self.net.parameters(), lr=self.inner_lr)
        if self.args.dataset == 'omniglot':
            self.loss = nn.CrossEntropyLoss()
        elif self.args.dataset == 'sine':
            self.loss = nn.MSELoss()

    def _train_epoch(self, x_spt, y_spt, x_qry, y_qry):
        """
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

        # i: num corrects and losses after i updates
        need_second_order = self.args.trainer == 'maml'

        corrects = [0 for _ in range(self.inner_step + 1)]
        losses_q = [0 for _ in range(self.inner_step + 1)]

        task_num, setsz, _, _, _ = x_spt.size()
        for t in range(task_num):
            # pre-update
            with torch.no_grad():
                loss_q, correct = self._evaluate(self.net.parameters(), x_qry[t], y_qry[t])
                losses_q[0] += loss_q
                corrects[0] += correct

            fast_weights = self.net.parameters()
            for i in range(self.inner_step):
                # the first update
                logits = self.net(x_spt[t], fast_weights)
                loss = self.loss(logits, y_spt[t])
                grad = torch.autograd.grad(loss, fast_weights, create_graph=need_second_order)
                fast_weights = list(map(lambda p, gradient: p - self.inner_lr * gradient, zip(fast_weights, grad)))

                loss_q, correct = self._evaluate(fast_weights, x_qry[t], y_qry[t])
                losses_q[i + 1] += loss_q
                corrects[i + 1] += correct

        # After all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # Update using meta optimizer
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()

        querysz = x_qry.size(1)
        accuracies = np.array(corrects) / (querysz * task_num)

        return accuracies

    def _evaluate(self, weights, x, y, net=None):
        if net is None:
            net = self.net
        logits = net(x, weights)
        loss = self.loss(logits, y)

        preds = F.softmax(logits, dim=1).argmax(dim=1)
        correct = torch.eq(preds, y).sum().item()

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
        corrects = [0 for _ in range(self.inner_step + 1)]
        losses_q = [0 for _ in range(self.inner_step + 1)]

        task_num, setsz, _, _, _ = x_spt.size()

        net = deepcopy(self.net)

        # pre-update
        with torch.no_grad():
            loss_q, correct = self._evaluate(net.parameters(), x_qry, y_qry)
            losses_q[0] += loss_q.item()
            corrects[0] += correct

        fast_weights = net.parameters()
        for i in range(self.inner_step_test):
            # the first update
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = self.loss(logits, y_spt)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p, gradient: p - self.inner_lr * gradient, zip(fast_weights, grad)))

            with torch.no_grad():
                loss_q, correct = self._evaluate(fast_weights, x_qry, y_qry, net=net)
                losses_q[i + 1] += loss_q.item()
                corrects[i + 1] += correct

        querysz = x_qry.size(0)
        accuracies = np.array(corrects) / querysz

        del net
        return accuracies
