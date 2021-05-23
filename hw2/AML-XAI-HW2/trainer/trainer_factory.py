import numpy as np
import torch

from utils import Saver


class TrainerFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_trainer(myModel, args):

        if args.trainer == 'maml' or args.trainer == 'fomaml':
            import trainer.maml as trainer
        elif args.trainer == 'reptile':
            import trainer.reptile as trainer

        return trainer.Trainer(myModel, args)


class GenericTrainer:
    '''
    Base class for trainer; to implement a new training routine, inherit from this. 
    '''

    def __init__(self, model, args):

        self.net = model
        self.args = args

        self.device = args.device
        # N way K shots
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num

        self.epoch = args.epoch
        self.inner_lr = args.inner_lr
        self.meta_lr = args.meta_lr
        self.inner_step = args.inner_step
        self.inner_step_test = args.inner_step_test
        self.output_path = args.output_path
        self.saver = Saver(self.output_path, args)

    def train(self, dataloader, epoch):
        eval_flag = True
        for step in range(epoch):
            x_spt, y_spt, x_qry, y_qry = dataloader.next()
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(self.device), torch.from_numpy(y_spt).to(
                self.device), \
                                         torch.from_numpy(x_qry).to(self.device), torch.from_numpy(y_qry).to(
                self.device)
            accs = self._train_epoch(x_spt, y_spt, x_qry, y_qry)
            eval_flag = False

            if step % 50 == 0:
                print('step:', step, '\ttraining acc:', accs)

            if step % 500 == 0:
                test_accs, test_losses = self.evaluate(dataloader)
                self.saver.save(self.net, test_accs, test_losses, step)

        test_accs, test_losses = self.evaluate(dataloader)
        self.saver.save(self.net, test_accs, test_losses, step)
        return test_accs, test_losses, self.net

    def evaluate(self, dataloader):
        test_accs = []
        test_losses = []
        for _ in range(1000 // self.task_num):
            # test
            x_spt, y_spt, x_qry, y_qry = dataloader.next('test')
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(self.device), torch.from_numpy(y_spt).to(
                self.device), torch.from_numpy(x_qry).to(self.device), torch.from_numpy(y_qry).to(self.device)

            # split to single task each time
            for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                test_acc, test_loss = self._finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                test_accs.append(test_acc)
                test_losses.append(test_loss)
        # [b, inner_step+1]
        test_accs = np.array(test_accs).mean(axis=0).astype(np.float16)
        test_losses = np.array(test_losses).mean(axis=0).astype(np.float16)
        print('Test acc:', test_accs)
        print('Test losses:', test_losses)
        return test_accs, test_losses

    def _train_epoch(self, x_spt, y_spt, x_qry, y_qry):
        pass

    def _finetunning(self, x_spt, y_spt, x_qry, y_qry):
        raise NotImplementedError
