import copy

import torch

'''
Hint: torch.autograd.grad function would be helpful to complete the following functions
'''


def simple_grad(model, x, y):
    model = copy.deepcopy(model)
    x = copy.deepcopy(x)

    model.eval()
    logits = model(x)
    logits_y = torch.gather(logits, 1, torch.unsqueeze(y, 1))
    logits_y.sum().backward()
    return x.grad.detach().cpu()


def smooth_grad(model, x, y, n_iter=10, alpha=0.1):
    ################### your answer should be written in here ################
    # dummy answer
    h = torch.zeros_like(x)

    ################### your answer should be written in here ################
    return h.detach().cpu()


def input_x_grad(model, x, y):
    ################### your answer should be written in here ################
    # dummy answer
    h = torch.zeros_like(x)

    ################### your answer should be written in here ################
    return h.detach().cpu()


def integrated_grad(model, x, y, x_b, n_iter=10):
    ################### your answer should be written in here ################
    # dummy answer
    h = torch.zeros_like(x)

    ################### your answer should be written in here ################
    return h.detach().cpu()


def grad_cam(model, x, y):
    target_layer_output = []

    def save_activation():
        def hook(model, input, output):
            target_layer_output.append(output)

        return hook

    layer, sub_layer = 'features', '28'
    hook_handler = model.__dict__['_modules'][layer][int(sub_layer)].register_forward_hook(save_activation())

    ################### your answer should be written in here ################
    # dummy answer
    h = torch.zeros_like(x)

    # hint : you would like to use function "torch.autograd.grad" to obtain
    #        a gradient of target class logit wrt a feature map 

    ################### your answer should be written in here ################
    hook_handler.remove()

    return h.detach().cpu()
