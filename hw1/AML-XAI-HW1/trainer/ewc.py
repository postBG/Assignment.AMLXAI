import torch
import torch.nn as nn
import torch.utils.data as td
from tqdm import tqdm

import trainer


def _init_fisher(model):
    diag_fisher = {}
    for name, param in model.named_parameters():
        diag_fisher[name] = param.clone().detach().fill_(0)
    return diag_fisher


def _get_named_params(model):
    params = {}
    for name, param in model.named_parameters():
        params[name] = param
    return params


def reg_loss(model, prev_model, importance):
    prev_params = _get_named_params(prev_model)
    loss = 0
    for name, param in model.named_parameters():
        loss += (importance[name] * (param - prev_params[name]) ** 2).sum()
    return loss


class Trainer(trainer.GenericTrainer):
    def __init__(self, model, args, optimizer, evaluator, task_info):
        super().__init__(model, args, optimizer, evaluator, task_info)

        self.lamb = args.lamb
        self.fisher = _init_fisher(self.model)
        self.ce_loss = nn.CrossEntropyLoss()

    def train(self, train_loader, test_loader, t, device=None):

        self.device = device
        lr = self.args.lr
        self.setup_training(lr)
        # Do not update self.t
        if t > 0:  # update fisher before starting training new task
            self.update_frozen_model()
            self.update_fisher()

        # Now, you can update self.t
        self.t = t

        self.train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=self.args.batch_size, shuffle=True)
        self.test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False)
        self.fisher_iterator = torch.utils.data.DataLoader(train_loader, batch_size=20, shuffle=True)

        for epoch in range(self.args.nepochs):
            self.model.train()
            self.update_lr(epoch, self.args.schedule)
            for data, target in tqdm(self.train_iterator):
                data, target = data.to(device), target.to(device)

                output = self.model(data)[t]
                loss_CE = self.criterion(output, target)

                self.optimizer.zero_grad()
                loss_CE.backward()
                self.optimizer.step()

            train_loss, train_acc = self.evaluator.evaluate(self.model, self.train_iterator, t, self.device)
            print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(epoch + 1, train_loss, 100 * train_acc),
                  end='')
            test_loss, test_acc = self.evaluator.evaluate(self.model, self.test_iterator, t, self.device)
            print(' Test: loss={:.3f}, acc={:5.1f}% |'.format(test_loss, 100 * test_acc), end='')
            print()

    def criterion(self, output, targets):
        """
        Arguments: output (The output logit of self.model), targets (Ground truth label)
        Return: loss function for the regularization-based continual learning
        
        For the hyperparameter on regularization, please use self.lamb
        """
        if self.t == 0:
            return self.ce_loss(output,targets)
        return self.ce_loss(output, targets) + self.lamb * reg_loss(self.model, self.model_fixed, self.fisher)

    def compute_diag_fisher(self):
        """
        Arguments: None. Just use global variables (self.model, self.criterion, ...)
        Return: Diagonal Fisher matrix. 
        
        This function will be used in the function 'update_fisher'
        """

        diag_fisher = _init_fisher(self.model_fixed)

        self.model.eval()
        for data, target in self.fisher_iterator:
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)[self.t]
            loss = self.ce_loss(output, target)
            loss.backward()

            for name, param in self.model.named_parameters():
                diag_fisher[name].data += param.grad.data ** 2 / len(self.fisher_iterator)

        return diag_fisher

    def update_fisher(self):

        """
        Arguments: None. Just use global variables (self.model, self.fisher, ...)
        Return: None. Just update the global variable self.fisher
        Use 'compute_diag_fisher' to compute the fisher matrix
        """
        recent_task_fisher = self.compute_diag_fisher()
        for name, value in recent_task_fisher.items():
            self.fisher[name].data += value.data
