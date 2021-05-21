import os

import numpy as np
import torch


class Saver(object):
    def __init__(self, output_path):
        self.output_path = output_path

        self.recent_acc = os.path.join(self.output_path, 'acc.txt')
        self.recent_loss = os.path.join(self.output_path, 'loss.txt')
        self.recent_model = os.path.join(self.output_path, 'model.pt')

        self.recent_best_acc = 0

    def save(self, net, test_accs, test_losses, step):
        self.save_best(net, test_accs, test_losses, step)
        self.save_recent(net, test_accs, test_losses)

    def save_best(self, net, test_accs, test_losses, step):
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
