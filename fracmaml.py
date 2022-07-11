import torch
from collections import OrderedDict
from torch.optim import Optimizer
from torch.nn import Module
from typing import Dict, List, Callable, Union
import numpy as np
from few_shot.core import create_nshot_task_label


def replace_grad(parameter_gradients, parameter_name):
    def replace_grad_(module):
        return parameter_gradients[parameter_name]

    return replace_grad_


def meta_gradient_step(model: Module,
                       optimiser: Optimizer,
                       loss_fn: Callable,
                       x: torch.Tensor,
                       y: torch.Tensor,
                       n_shot: int,
                       k_way: int,
                       q_queries: int,
                       order: int,
                       inner_train_steps: int,
                       inner_lr: float,
                       train: bool,
                       device: Union[str, torch.device]):
    """
    Perform a gradient step on a meta-learner.

    # Arguments
        model: Base model of the meta-learner being trained
        optimiser: Optimiser to calculate gradient step from loss
        loss_fn: Loss function to calculate between predictions and outputs
        x: Input samples for all few shot tasks
        y: Input labels of all few shot tasks
        n_shot: Number of examples per class in the support set of each task
        k_way: Number of classes in the few shot classification task of each task
        q_queries: Number of examples per class in the query set of each task. The query set is used to calculate
            meta-gradients after applying the update to
        order: Whether to use 1st order MAML (update meta-learner weights with gradients of the updated weights on the
            query set) or 2nd order MAML (use 2nd order updates by differentiating through the gradients of the updated
            weights on the query with respect to the original weights).
        inner_train_steps: Number of gradient steps to fit the fast weights during each inner update
        inner_lr: Learning rate used to update the fast weights on the inner update
        train: Whether to update the meta-learner weights at the end of the episode.
        device: Device on which to run computation
    """
    data_shape = x.shape[2:]
    create_graph = (True if order == 2 else False) and train

    task_gradients = []
    task_losses = []
    task_predictions = []
    index_meta_batch=0
    for meta_batch in x:
        # By construction x is a 5D tensor of shape: (meta_batch_size, n*k + q*k, channels, width, height)
        # Hence when we iterate over the first  dimension we are iterating through the meta batches
        x_task_train = meta_batch[:n_shot * k_way]
        x_task_val = meta_batch[n_shot * k_way:]

        # Create a fast model using the current meta model weights
        fast_weights = OrderedDict(model.named_parameters())

        # Train the model for `inner_train_steps` iterations
        for inner_batch in range(inner_train_steps):
            # Perform update of model weights
            y = create_nshot_task_label(k_way, n_shot).to(device)
            logits = model.functional_forward(x_task_train, fast_weights)
            loss = loss_fn(logits, y)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)

            # Update weights manually
            fast_weights = OrderedDict(
                (name, param - inner_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )
            if inner_batch<=0:
                grad_list_inner = gradients
            else:
                grad_list_inner=np.row_stack((grad_list_inner, gradients))
        
        
        # Do a pass of the model on the validation data from the current task
        y = create_nshot_task_label(k_way, q_queries).to(device)
        logits = model.functional_forward(x_task_val, fast_weights)
        loss = loss_fn(logits, y)
        loss.backward(retain_graph=True)

        # Get post-update accuracies
        y_pred = logits.softmax(dim=1)
        task_predictions.append(y_pred)

        # Accumulate losses and gradients
        task_losses.append(loss)
        gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)

        
        

        grad_list = np.row_stack((grad_list_inner, gradients))
        # grad_inner_frac=grad_list[-1]+(-0.9*grad_list[-2])+(-0.0450*grad_list[-3])+(-0.0165*grad_list[-4])+(-0.008662*grad_list[-5])  ###frac=0.9    h=1
        # grad_inner_frac=grad_list[-1]+(-0.482298*grad_list[-3])+(-0.0129228*grad_list[-5])    ###frac=0.9    h=2
        # grad_inner_frac=grad_list[-1]+(-0.17411*grad_list[-3])+(-0.060628*grad_list[-5])   ###frac=0.2    h=2
        # grad_inner_frac=grad_list[-1]+(-0.16054*grad_list[-3])+(-0.051551*grad_list[-5])   ###frac=0.2    h=3


        # FracMAML1   -0.1   1             result  1.24780425
        # grad_inner_frac=(1*grad_list[-1])+0.1*grad_list[-2]+(0.055*grad_list[-3])+(0.0385*grad_list[-4])+(0.0298375*grad_list[-5])+(0.02446675*grad_list[-6])

        # FracMAML2   -0.2    1          result    1.537436
        # grad_inner_frac=(1*grad_list[-1])+0.1999*grad_list[-2]+(0.12*grad_list[-3])+(0.088*grad_list[-4])+(0.0704*grad_list[-5])+(0.059136*grad_list[-6])

        # FracMAML3   -0.3    1            result     1.8739076
        # grad_inner_frac=(1*grad_list[-1])+(0.3*grad_list[-2])+0.195*grad_list[-3]+(0.1495*grad_list[-4])+(0.1233374*grad_list[-5])+(0.1060702*grad_list[-6])

        # FracMAML4    -0.2    2           result    1.521453
        # grad_inner_frac=(1*grad_list[-1])+0.22973*grad_list[-2]+(0.15834*grad_list[-4])+(0.1333830*grad_list[-6])

        # FracMAML5    -0.1   2          result          1.21775475
        # grad_inner_frac=(1*grad_list[-1])+0.1071773*grad_list[-2]+(0.0631784*grad_list[-4])+(0.04739905*grad_list[-6])

        # FracMAML6    -0.4   1          result       2.2619309
        # grad_inner_frac=(1*grad_list[-1])+(0.399999*grad_list[-2])+0.2799999*grad_list[-3]+(0.22399*grad_list[-4])+(0.19039*grad_list[-5])+(0.167552*grad_list[-6])


        # FracMAML7    -0.3   2           result     1.94388
        # grad_inner_frac=(1*grad_list[-1])+0.36934*grad_list[-2]+(0.295564*grad_list[-4])+(0.278976*grad_list[-6])


        # FracMAML8    -0.1   3           result        1.180115
        # grad_inner_frac=(1*grad_list[-1])+0.1116*grad_list[-2]+(0.068515*grad_list[-5])


        # FracMAML9    -0.2   3              result        1.4353671
        grad_inner_frac=(1*grad_list[-1])+0.2491461*grad_list[-2]+(0.186221*grad_list[-5])

        # FracMAML10    -0.3   3              result      1.794086498
        # grad_inner_frac=(1*grad_list[-1])+0.417116*grad_list[-2]+(0.376970498*grad_list[-5])



        # FracMAML11    -0.05   1              result      1.10786484
        # grad_inner_frac=grad_list[-1]+0.417116*grad_list[-2]+0.376970498*grad_list[-3]+0.417116*grad_list[-4]+0.376970498*grad_list[-5]+0.417116*grad_list[-6]




        # grad_inner_frac=0.199999*grad_list[-1]+(0.12*grad_list[-2])+(0.088*grad_list[-3])+(0.0704*grad_list[-4])+(0.059136*grad_list[-5])+(0.05125*grad_list[-6])   ###frac=-0.2    h=1
        
        
        # grad_inner_frac=grad_list[-1]      ###ceshi
        if index_meta_batch<=0:
            grad_list_final = grad_inner_frac
        else:
            grad_list_final = np.row_stack((grad_list_final, grad_inner_frac))
        index_meta_batch=index_meta_batch+1
        
        named_grads = {name: g for ((name, _), g) in zip(fast_weights.items(), gradients)}
        task_gradients.append(named_grads)    
        

    sum_grads_pi=grad_list_final.mean(axis=0)


    # grad_final1=sum_grads_pi
        
# 1
# -0.9
# -0.045000
# -0.0165
# -0.008662
# -0.00537075
# -0.0036700
# -0.00267
# -0.00203
 # -0.0016084
 # -0.00130280
        
    if order == 1:
        if train:
            sum_task_gradients = {k: torch.stack([grad[k] for grad in task_gradients]).mean(dim=0)
                                  for k in task_gradients[0].keys()}
            named_grads_final = {name: g for ((name, _), g) in zip(fast_weights.items(), sum_grads_pi)}
            hooks = []
            for name, param in model.named_parameters():
                hooks.append(
                    param.register_hook(replace_grad(named_grads_final, name))
                )

            model.train()
            optimiser.zero_grad()
            # Dummy pass in order to create `loss` variable
            # Replace dummy gradients with mean task gradients using hooks
            logits = model(torch.zeros((k_way, ) + data_shape).to(device, dtype=torch.double))
            loss = loss_fn(logits, create_nshot_task_label(k_way, 1).to(device))
            loss.backward()
            optimiser.step()

            for h in hooks:
                h.remove()

        return torch.stack(task_losses).mean(), torch.cat(task_predictions)

    elif order == 2:
        model.train()
        optimiser.zero_grad()
        meta_batch_loss = torch.stack(task_losses).mean()

        if train:
            meta_batch_loss.backward()
            optimiser.step()

        return meta_batch_loss, torch.cat(task_predictions)
    else:
        raise ValueError('Order must be either 1 or 2.')
