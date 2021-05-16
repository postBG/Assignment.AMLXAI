import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import optim
import numpy as np

from copy import deepcopy
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
        corrects = [0 for _ in range(self.inner_step + 1)]
        losses_q = [0 for _ in range(self.inner_step + 1)]

        task_num, setsz, _, _, _ = x_spt.size()
        for i in range(task_num):
            # on the first example (k=0)
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = self.loss(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p, gradient: p - self.inner_lr * gradient,
                                    zip(self.net.parameters(), grad)))

            # pre-update
            k = 0
            with torch.no_grad():
                loss_q, correct = self._evaluate(self.net.parameters(), x_qry[i], y_qry[i])
                losses_q[k] += loss_q
                corrects[k] += correct

            # after the first update
            k = 1
            with torch.no_grad():
                loss_q, correct = self._evaluate(fast_weights, x_qry[i], y_qry[i])
                losses_q[k] += loss_q
                corrects[k] += correct

            for k in range(1, self.inner_step):
                # 1. run the i-th task and compute loss for k=1 ~ K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = self.loss(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p, gradient: p - self.inner_lr * gradient, zip(fast_weights, grad)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = self.loss(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # Update using meta optimizer
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()

        querysz = x_qry.size(1)
        accs = np.array(corrects) / (querysz * task_num)

        return accs

    def _evaluate(self, weights, x, y):
        logits = self.net(x, weights, bn_training=True)
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
        results = [0 for _ in range(self.inner_step + 1)]

        ##########################################################################################

        # Write your code here

        ##########################################################################################

        return results
