import torch
import torch.nn as nn
import torch.utils.data as td
from tqdm import tqdm

import trainer


def _get_named_params(model):
    params = {}
    for name, param in model.named_parameters():
        params[name] = param
    return params


def l2_loss(model, prev_model):
    prev_params = _get_named_params(prev_model)
    reg_loss = 0
    for name, param in model.named_parameters():
        reg_loss += ((param - prev_params[name]) ** 2).sum()
    return reg_loss


class Trainer(trainer.GenericTrainer):
    def __init__(self, model, args, optimizer, evaluator, taskcla):
        super().__init__(model, args, optimizer, evaluator, taskcla)

        self.lamb = args.lamb
        self.ce_loss = nn.CrossEntropyLoss()

    def train(self, train_loader, test_loader, t, device=None):

        self.device = device
        lr = self.args.lr
        self.setup_training(lr)
        # Do not update self.t
        if t > 0:
            self.update_frozen_model()

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
                batch_size = data.shape[0]

                output = self.model(data)[t]
                loss_CE = self.criterion(output, target)

                self.optimizer.zero_grad()
                loss_CE.backward()
                self.optimizer.step()

            train_loss, train_acc = self.evaluator.evaluate(self.model, self.train_iterator, t, self.device)
            num_batch = len(self.train_iterator)
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
        return self.ce_loss(output, targets) + self.lamb * l2_loss(self.model, self.model_fixed)

