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
    # Treat the alpha as the variance
    model = copy.deepcopy(model)
    x = copy.deepcopy(x)

    for i in range(n_iter):
        noises = torch.randn_like(x) * alpha
        logits = model(x + noises)
        logits_y = torch.gather(logits, 1, torch.unsqueeze(y, 1))
        logits_y.sum().backward()

    h = x.grad.detach().cpu() / n_iter
    return h


def input_x_grad(model, x, y):
    grad = simple_grad(model, x, y)
    return x.detach().cpu() * grad


def integrated_grad(model, x, y, x_b, n_iter=10):
    model = copy.deepcopy(model)
    x = copy.deepcopy(x)
    x_b = copy.deepcopy(x_b)

    paths = [x_b + (i / n_iter) * (x - x_b) for i in range(n_iter + 1)]
    for p in paths:
        logits = model(p)
        logits_y = torch.gather(logits, 1, torch.unsqueeze(y, 1))
        logits_y.sum().backward()

    h = (x - x_b) * x.grad
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
