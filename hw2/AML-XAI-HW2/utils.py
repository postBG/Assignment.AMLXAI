import os

import numpy as np
import torch


class Saver(object):
    def __init__(self, output_path):
        self.output_path = output_path

        self.recent_acc = os.path.join(self.output_path, 'acc.txt')
        self.recent_loss = os.path.join(self.output_path, 'loss.txt')
        self.recent_model = os.path.join(self.output_path, 'model.pt')

        self.best_acc = os.path.join(self.output_path, 'best_acc.txt')
        self.best_loss = os.path.join(self.output_path, 'best_loss.txt')
        self.best_model = os.path.join(self.output_path, 'best_model.pt')

        self.recent_best_acc = 0

    def save(self, net, test_accs, test_losses):
        self.save_best(net, test_accs, test_losses)
        self.save_recent(net, test_accs, test_losses)

    def save_best(self, net, test_accs, test_losses):
        recent_acc = test_accs[-1]
        if recent_acc <= self.recent_best_acc:
            return

        self.recent_best_acc = recent_acc
        print('Save Best at ' + self.output_path)
        np.savetxt(self.best_acc, test_accs, '%.4f')
        np.savetxt(self.best_loss, test_losses, '%.4f')
        torch.save(net.state_dict(), self.best_model)
        print('done!')

    def save_recent(self, net, test_accs, test_losses):
        print('Save Recent at ' + self.output_path)
        np.savetxt(self.recent_acc, test_accs, '%.4f')
        np.savetxt(self.recent_loss, test_losses, '%.4f')
        torch.save(net.state_dict(), self.recent_model)
        print('done!')
