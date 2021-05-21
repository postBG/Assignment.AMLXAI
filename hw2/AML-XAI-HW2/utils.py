import os

import numpy as np
import torch


def save(net, output_path, test_accs, test_losses):
    acc_path = os.path.join(output_path, 'acc.txt')
    loss_path = os.path.join(output_path, 'loss.txt')
    model_path = os.path.join(output_path, 'task.pt')
    print('Save at ' + output_path)
    np.savetxt(acc_path, test_accs, '%.4f')
    np.savetxt(loss_path, test_losses, '%.4f')
    torch.save(net.state_dict(), model_path)
    print('done!')
