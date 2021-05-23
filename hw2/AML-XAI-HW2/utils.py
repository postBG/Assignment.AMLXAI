import os

import numpy as np
import torch


class Saver(object):
    def __init__(self, output_path, args):
        self.output_path = output_path
        self.args = args

        self.recent_acc = os.path.join(self.output_path, 'acc.txt')
        self.recent_loss = os.path.join(self.output_path, 'loss.txt')
        self.recent_model = os.path.join(self.output_path, 'model.pt')

        self.use_loss = (self.args.dataset == 'sine')
        self.recent_best_loss = np.inf
        self.recent_best_acc = 0

    def save(self, net, test_accs, test_losses, step):
        if self.use_loss:
            self.save_best_loss(net, test_accs, test_losses, step)
        else:
            self.save_best_acc(net, test_accs, test_losses, step)
        self.save_recent(net, test_accs, test_losses)

    def save_best_loss(self, net, test_accs, test_losses, step):
        recent_loss = test_losses[-1]
        if self.recent_best_loss < recent_loss:
            return

        best_acc = os.path.join(self.output_path, f'best_acc_{step}.txt')
        best_loss = os.path.join(self.output_path, f'best_loss_{step}.txt')
        best_model = os.path.join(self.output_path, 'best_model.pt')

        self.recent_best_loss = recent_loss
        print('Save Best at ' + self.output_path)
        np.savetxt(best_acc, test_accs, '%.4f')
        np.savetxt(best_loss, test_losses, '%.4f')
        torch.save(net.state_dict(), best_model)
        print('done!')

    def save_best_acc(self, net, test_accs, test_losses, step):
        recent_acc = test_accs[-1]
        if recent_acc <= self.recent_best_acc:
            return

        best_acc = os.path.join(self.output_path, f'best_acc_{step}.txt')
        best_loss = os.path.join(self.output_path, f'best_loss_{step}.txt')
        best_model = os.path.join(self.output_path, 'best_model.pt')

        self.recent_best_acc = recent_acc
        print('Save Best at ' + self.output_path)
        np.savetxt(best_acc, test_accs, '%.4f')
        np.savetxt(best_loss, test_losses, '%.4f')
        torch.save(net.state_dict(), best_model)
        print('done!')

    def save_recent(self, net, test_accs, test_losses):
        print('Save Recent at ' + self.output_path)
        np.savetxt(self.recent_acc, test_accs, '%.4f')
        np.savetxt(self.recent_loss, test_losses, '%.4f')
        torch.save(net.state_dict(), self.recent_model)
        print('done!')
