"""
Implementation of Dense and Convolutional Bayesian layers employing LWTA activations and IBP in pyTorch, as described in
Panousis et al., Nonparametric Bayesian Deep Networks with Local Competition.
ReLU and Linear activations are also implemented.

@author: Konstantinos P. Panousis
Cyprus University of Technology
"""

import torch, math
from torch.nn import Module, Parameter
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.distributions.one_hot_categorical import OneHotCategorical as Categorical

"""
Implementation of utility functions for the Dense and Convolutional Bayesian layers employing LWTA activations and IBP in pyTorch, as described in
Panousis et al., Nonparametric Bayesian Deep Networks with Local Competition.

@author: Konstantinos P. Panousis
Cyprus University of Technology
"""


###################################################
################## DIVERGENCES ####################
###################################################

def kl_divergence_kumaraswamy(prior_a, a, b):
    """
    KL divergence for the Kumaraswamy distribution.

    :param prior_a: torch tensor: the prior a concentration
    :param prior_b: torch tensor: the prior b concentration
    :param a: torch tensor: the posterior a concentration
    :param b: torch tensor: the posterior b concentration
    :param sample: a sample from the Kumaraswamy distribution

    :return: scalar: the kumaraswamy kl divergence
    """

    Euler = torch.tensor(0.577215664901532)
    kl = (1 - prior_a / a) * (-Euler - torch.digamma(b) - 1. / b) \
         + torch.log(a * b / prior_a) - (b - 1) / b

    return kl.sum()


def kl_divergence_normal(prior_mean, prior_scale, posterior_mean, posterior_scale):
    """
     Compute the KL divergence between two Gaussian distributions.

    :param prior_mean: torch tensor: the mean of the prior Gaussian distribution
    :param prior_scale: torch tensor: the scale of the prior Gaussian distribution
    :param posterior_mean: torch tensor: the mean of the posterior Gaussian distribution
    :param posterior_scale: torch tensor: the scale of the posterior Gaussian distribution

    :return: scalar: the kl divergence between the prior and posterior distributions
    """

    device = torch.device("cuda" if posterior_mean.is_cuda else "cpu")

    prior_scale_normalized = F.softplus(torch.Tensor([prior_scale]).to(device))
    posterior_scale_normalized = F.softplus(posterior_scale)

    kl_loss = -0.5 + torch.log(prior_scale_normalized) - torch.log(posterior_scale_normalized) \
              + (posterior_scale_normalized ** 2 + (posterior_mean - prior_mean) ** 2) / (
                          2 * prior_scale_normalized ** 2)

    return kl_loss.sum()


def model_kl_divergence_loss(model, kl_weight=1.):
    """
    Compute the KL losses for all the layers of the considered model.

    :param model: nn.Module extension implementing the model with our custom layers
    :param kl_weight: scalar: the weight for the KL divergences

    :return: scalar: the KL divergence for all the layers of the model.
    """

    device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
    kl_sum = torch.Tensor([0]).to(device)

    # get the layers as a list
    model_children = list(model.children())

    for layer in model_children:
        if hasattr(layer, 'loss'):
            kl_sum += layer.loss

    return kl_weight * kl_sum[0]


###########################################
########## DISTRIBUTIONS ##################
###########################################


def kumaraswamy_sample(conc1, conc0, sample_shape, eps = 1e-8):
    """
    Sample from the Kumaraswamy distribution given the concentrations

    :param conc1: torch tensor: the a concentration of the distribution
    :param conc0: torch tensor: the b concentration of the distribution
    :param batch_shape: scalar: the batch shape for the samples

    :return: torch tensor: a sample from the Kumaraswamy distribution
    """

    U = torch.rand(sample_shape, device = conc1.device)
    x = (1. - (1. - 1e-6)) * U + (1. - 1e-6)
    q_u = (x ** (1. / conc0)) ** (1. / conc1)

    return q_u


def bin_concrete_sample(a, temperature, hard=False, eps=1e-8):
    """"
    Sample from the binary concrete distribution
    """

    U = torch.rand(a.shape, device = a.device)
    L = torch.log(U + eps) - torch.log(1. - U + eps)
    X = torch.sigmoid((L + a) / temperature)

    return X


def concrete_sample(a, temperature, hard=False, eps=1e-8, axis=-1):
    """
    Sample from the concrete relaxation.
    :param probs: torch tensor: probabilities of the concrete relaxation
    :param temperature: float: the temperature of the relaxation
    :param hard: boolean: flag to draw hard samples from the concrete distribution
    :param eps: float: eps to stabilize the computations
    :param axis: int: axis to perform the softmax of the gumbel-softmax trick
    :return: a sample from the concrete relaxation with given parameters
    """

    U = torch.rand(a.shape, device = a.device)
    G = - torch.log(- torch.log(U + eps) + eps)
    t = (a + G) / temperature

    y_soft = F.softmax(t, axis)

    if hard:
        _, k = y_soft.data.max(axis)
        shape = y_soft.size()

        if len(a.shape) == 2:
            y_hard = y_soft.new(*shape).zero_().scatter_(-1, k.view(-1, 1), 1.0)
        else:
            y_hard = y_soft.new(*shape).zero_().scatter_(-1, k.view(-1, 1, a.size(1), a.size(2), a.size(3)), 1.0)

        y = (y_hard - y_soft).detach() + y_soft

    else:
        y = y_soft

    return y



###############################################
################ CONSTRAINTS ##################
###############################################
class parameterConstraints(object):
    """
    A class implementing the constraints for the parameters of the layers.
    """

    def __init__(self):
        pass

    def __call__(self, module):
        if hasattr(module, 'posterior_un_scale'):
            scale = module.posterior_un_scale
            scale = scale.clamp(-7., 1000.)
            module.posterior_un_scale.data = scale

        if hasattr(module, 'bias_un_scale'):
            scale = module.bias_un_scale
            scale = scale.clamp(-7., 1000.)
            module.bias_un_scale.data = scale

        if hasattr(module, 'conc1') and module.conc1 is not None:
            conc1 = module.conc1
            conc1 = conc1.clamp(-6., 1000.)
            module.conc1.data = conc1

        if hasattr(module, 'conc0') and module.conc0 is not None:
            conc0 = module.conc0
            conc0 = conc0.clamp(-6., 1000.)
            module.conc0.data = conc0

        if hasattr(module, 't_pi') and module.t_pi is not None:
            t_pi = module.t_pi
            t_pi = t_pi.clamp(-7, 600.)
            module.t_pi.data = t_pi


class LWTA(nn.Module):
    """
    A simple implementation of the LWTA activation to be used as a standalone approach for various models.
    """

    def __init__(self, inplace = True, U=2):
        super(LWTA, self).__init__()
        self.inplace = inplace
        self.temp = .67
        self.temp_test = 0.01
        self.kl_ = 0.
        self.U = U

    def forward(self, input):
        out, kl = lwta_activation(input, U = self.U, training = self.training,
                                  temperature = self.temp,
                                  temp_test = self.temp_test)
        self.kl_ = kl

        return out

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


def lwta_activation(input, U = 2, training = True, temperature = 0.67, temp_test = 0.01):
    """
    The general LWTA activation function.
    Can be either deterministic or stochastic depending on the input.
    Deals with both FC and Conv Layers.
    """
    out = input.clone()
    kl= 0.

    # case of a fully connected layer
    if len(out.shape) == 2:
        logits = torch.reshape(out, [-1, out.size(1)//U, U])
        
        if training:
            mask = concrete_sample( logits, 0.67)
        else:
            mask = concrete_sample(logits, temp_test  )
        mask_r = mask.reshape(input.shape)
    else:
        x = torch.reshape(out, [-1, out.size(1)//U, U, out.size(-2), out.size(-1)])
        
        logits = x
        if training:
            mask = concrete_sample(logits, temperature, axis = 2)
        else:
            mask = concrete_sample(logits , temp_test, axis = 2)

        mask_r = mask.reshape(input.shape)

    if training:
        q = mask
        log_q = torch.log(q + 1e-8)
        log_p = torch.log(torch.tensor(1.0 / U))

        kl = torch.sum(q * (log_q - log_p), 1)
        kl = torch.mean(kl) / 1000.

    input *= mask_r

    return input, kl


