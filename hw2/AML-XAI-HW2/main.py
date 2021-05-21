import os
import random
import datetime

import numpy as np
import torch

import data_handler
import networks
import trainer
from argument import get_args


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(args)

    if not os.path.isdir('result_data'):
        print('Make directory for saving results')
        os.makedirs('result_data')

    now = datetime.datetime.now()
    log_name = '{}_{}_{}way_{}shot_{}query_{}tasks_mlr{}_ilr{}_is{}_ist{}_epoch{}_{}'.format(
        args.trainer, args.dataset, args.n_way, args.k_spt, args.k_qry, args.task_num, args.meta_lr,
        args.inner_lr, args.inner_step, args.inner_step_test, args.epoch, now.strftime('%Y-%m-%d_%H_%M_%S'))
    output_path = './result_data/' + log_name
    args.output_path = output_path

    if not os.path.isdir(os.path.join(output_path)):
        print('Make directory for saving results')
        os.makedirs(output_path)

    device = torch.device('cuda')

    args.device = device
    myModel = networks.ModelFactory.get_model(args.dataset, args.n_way).to(device)

    myTrainer = trainer.TrainerFactory.get_trainer(myModel, args)
    dataloader = data_handler.get_dataset(args)
    test_accs, test_losses, net = myTrainer.train(dataloader, args.epoch)


if __name__ == '__main__':
    main()
