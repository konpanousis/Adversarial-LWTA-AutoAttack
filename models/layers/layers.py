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

class DenseBayesian(Module):
    """
    Class for a Bayesian Dense Layer employing various activations, namely ReLU, Linear and LWTA, along with IBP.
    """

    def __init__(self, input_features, output_features, competitors,
                 activation = 'lwta', deterministic = True, temperature = 0.67,
                 ibp = False, bias=True):
        """

        :param input_features: int: the number of input_features
        :param output_features: int: the number of output features
        :param competitors: int: the number of competitors in case of LWTA activations, else 1
        :param activation: str: the activation to use. 'lwta', 'linear' or 'lwta'
        :param prior_mean: float: the prior mean for the gaussian distribution
        :param prior_scale: float: the prior scale for the gaussian distribution
        :param temperature: float: the temperature of the relaxations.
        :param ibp: boolean: flag to use the IBP prior.
        :param bias: boolean: flag to use bias.
        """

        super(DenseBayesian, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.K = output_features // competitors
        self.U = competitors
        self.activation = activation
        self.deterministic = deterministic

        self.temperature = temperature
        self.ibp = ibp
        self.bias  = bias
        self.tau = 1e-2
        #self.training = True


        #################################
        #### DEFINE THE PARAMETERS ######
        #################################

        self.posterior_mean = Parameter(torch.Tensor(output_features, input_features))

        if not deterministic:
            # posterior unnormalized scale. Needs to be passed by softplus
            self.posterior_un_scale = Parameter(torch.Tensor(output_features, input_features))
            self.register_buffer('weight_eps', None)

        if activation == 'lwta' or activation == 'lwta_binary' or activation == 'hard_lwta':
            if competitors == 1:
                print('Cant perform competition with 1 competitor.. Setting to default: 2\n')
                self.U = 2
                self.K = output_features // 2
            if output_features % self.U != 0:
                raise ValueError('Incorrect number of competitors. '
                                 'Cant divide {} units in groups of {}..'.format(output_features, competitors))
        else:
            if competitors != 1:
                print('Wrong value of competitors for activation ' + activation + '. Setting value to 1.')
                self.K = output_features
                self.U = 1

        #########################
        #### IBP PARAMETERS #####
        #########################
        if ibp:
            self.prior_conc1 = torch.tensor(1.)
            self.prior_conc0 = torch.tensor(1.)

            self.conc1 = Parameter(torch.Tensor(self.K))
            self.conc0 = Parameter(torch.Tensor(self.K))

            self.t_pi = Parameter(torch.Tensor(input_features, self.K))
        else:
            self.register_parameter('prior_conc1', None)
            self.register_parameter('prior_conc0', None)
            self.register_parameter('conc1', None)
            self.register_parameter('conc0', None)
            self.register_parameter('t_pi', None)

        if bias:
            self.bias_mean = Parameter(torch.Tensor(output_features))

            if not deterministic:
                self.bias_un_scale = Parameter(torch.Tensor(output_features))
                self.register_buffer('bias_eps', None)
        else:
            self.register_parameter('bias_mean', None)
            if not deterministic:
                self.register_parameter('bias_un_scale', None)
                self.register_buffer('bias_eps', None)

        if self.ibp:
            self.ibp_params = nn.ParameterList([self.conc1, self.conc0, self.t_pi])

        self.params = nn.ParameterList([self.posterior_mean])
        if bias:
            self.params.extend([self.bias_mean])
        if not deterministic:
            self.params.append(self.bias_un_scale)
            self.params.append(self.posterior_un_scale)
        self.reset_parameters()


    def reset_parameters(self):
        """
        Initialization function for all the parameters of the Dense Bayesian Layer.

        :return: null
        """


        # original init
        init.xavier_uniform_(self.posterior_mean)

        if not self.deterministic:
            self.posterior_un_scale.data.fill_(-5.)

        if self.bias:
            self.bias_mean.data.fill_(0.)

            if not self.deterministic:
                self.bias_un_scale.data.fill_(-5.)

        if self.ibp:
            self.conc1.data.fill_(self.K)
            self.conc0.data.fill_(2.)

            init.uniform_(self.t_pi, 4., 5.)


    def forward(self, input):
        """
        Override the default forward function to implement the Bayesian layer.

        :param input: torch tensor: the input data

        :return: torch tensor: the output of the current layer
        """

        layer_loss = 0.
        if self.training:

            if not self.deterministic:
                # use the reparameterization trick
                posterior_scale = F.softplus(self.posterior_un_scale)
                W = self.posterior_mean + posterior_scale * torch.randn_like(self.posterior_un_scale)

                kl_weights = -0.5 * torch.mean(2*posterior_scale - torch.square(self.posterior_mean)
                                               - posterior_scale ** 2 + 1)
                layer_loss += torch.sum(kl_weights)

            else:
                W = self.posterior_mean


            if self.ibp:
                z, kl_sticks, kl_z = self.indian_buffet_process(self.temperature)

                W = z.T*W

                layer_loss += kl_sticks
                layer_loss += kl_z

            if self.bias:
                if not self.deterministic:
                    bias = self.bias_mean + F.softplus(self.bias_un_scale) * torch.randn_like(self.bias_un_scale)
                else:
                    bias = self.bias_mean
            else:
                bias = None

        else:

            W = self.posterior_mean

            if self.bias:
                bias = self.bias_mean
            else:
                bias = None

            if self.ibp:
                z, _, _ = self.indian_buffet_process(0.67)
                W = z.T*W

        out = F.linear(input, W, bias)

        # apply the given activation function
        if self.activation == 'linear':
            self.kl_ = layer_loss
            return out

        elif self.activation == 'relu':
            self.kl_ = layer_loss
            self.acts = out
            return F.relu(out)

        elif self.activation == 'elu':
            self.kl_ = layer_loss
            return F.elu(out)

        elif self.activation == 'lwta':
            out, kl =  self.lwta_activation(out, self.temperature if self.training else 0.01)
            layer_loss += kl
            self.kl_ = layer_loss

            return out

        elif self.activation == 'hard_lwta':
            self.kl_ = layer_loss

            logits = torch.reshape(out, [-1, out.size(1) // self.U, self.U])
            a = torch.argmax(logits, 2, keepdims=True)
            mask = torch.zeros_like(logits).scatter_(2, a, 1.).reshape(out.shape)

            out = out*mask

            return out

        elif self.activation == 'lwta_binary':
            out, kl = self.lwta_activation(out, self.temperature if self.training else 0.01,
                                           binary = True)

            layer_loss += kl
            self.kl_ = layer_loss

            return out

        else:
            raise ValueError(self.activation + " is not implemented..")


    def indian_buffet_process(self, temp = 0.67):
        """
            Compute the KL divergence for the infian buffet process imposed on the synapses
            of a fully connected layer.
            If we are in eval mode, we utilize the cut-off threshold to remove connections from the architecture.

        """

        kl_sticks = kl_z = 0.
        z_sample = bin_concrete_sample(self.t_pi, temp)

        if not self.training:
            t_pi_sigmoid = torch.sigmoid(self.t_pi)
            mask = t_pi_sigmoid > self.tau
            z_sample = t_pi_sigmoid*mask

        z = z_sample.repeat(1, self.U)

        # compute the KL terms only during training
        if self.training:

            a_soft = F.softplus(self.conc1)
            b_soft = F.softplus(self.conc0)

            q_u = kumaraswamy_sample(a_soft, b_soft, sample_shape = [self.t_pi.size(0), self.t_pi.size(1)])
            prior_pi = torch.cumprod(q_u, -1)

            q = torch.sigmoid(self.t_pi)
            log_q = torch.log(q + 1e-6)
            log_p = torch.log(prior_pi + 1e-6)

            kl_z = torch.sum(q*(log_q - log_p))
            kl_sticks = torch.sum(kl_divergence_kumaraswamy(torch.ones_like(a_soft), a_soft, b_soft))

        return z, kl_sticks, kl_z


    def lwta_activation(self, input, temp = 0.67, binary = False):
        """
        Function implementing the LWTA activation with a competitive random sampling procedure as described in
        Panousis et al., Nonparametric Bayesian Deep Networks with Local Competition.

        :param temp: float: the temperature of the relaxation
        :param binary: boolean: flag to draw hard samples from the binary relaxations.
        :param input: torch tensor: the input to compute the LWTA activations.

        :return: torch tensor: the output after the lwta activation
        """

        kl = 0.

        logits = torch.reshape(input, [-1, self.K, self.U])
        if self.training:
            xi = concrete_sample(logits, temperature = temp)
        else:
            xi = concrete_sample(logits, temp, axis=-1)

        self.probs = F.softmax(logits, -1)

        if binary:
            out = xi

        else:
            out = logits*xi

        out = out.reshape(input.shape)

        # compute the KL divergence during training
        if self.training:
            q = xi
            log_q = torch.log(q + 1e-8)
            log_p = torch.log(torch.tensor(1.0/self.U))

            kl = torch.sum(q*(log_q - log_p),1)
            kl = torch.mean(kl)/1000.

        return out, kl



###########################################################################
##################### BAYESIAN CONV IMPLEMENTATION ########################
###########################################################################
class ConvBayesian(Module):
    """
    Class for a Bayesian Conv Layer employing various activations, namely ReLU, Linear and LWTA, along with IBP.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding =0 , competitors =2,
                 activation = 'lwta', deterministic = True, temperature = 0.67, ibp = False, bias=True,
                 batch_norm = False):
        """

        :param in_channels: int: the number of input channels
        :param out_channels: int: the number of the output channels
        :param kernel_size: int: the size of the kernel
        :param stride: int: the stride of the kernel
        :param padding: int: padding for the convolution operation
        :param competitors: int: the number of competitors in case of LWTA activations, else 1
        :param activation: str: the activation to use. 'relu', 'linear' or 'lwta'
        :param prior_mean: float: the prior mean for the gaussian distribution
        :param prior_scale: float: the prior scale for the gaussian distribution
        :param temperature: float: the temperature for the employed relaxations
        :param ibp: boolean: flag to use the IBP prior
        :param bias: boolean; flag to use bias
        """

        super(ConvBayesian, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.K = out_channels // competitors
        self.U = competitors
        self.activation = activation
        self.deterministic = deterministic

        self.temperature = temperature
        self.ibp = ibp
        self.bias  = bias
        self.tau = 1e-2
        self.batch_norm = batch_norm

        if self.batch_norm:
            self.bn_layer = nn.BatchNorm2d(out_channels)

        if activation == 'lwta' or activation == 'hard_lwta':
            self.temp = Parameter(torch.tensor(temperature))
            if competitors == 1:
                print('Cant perform competition with 1 competitor.. Setting to default: 2')
                self.U = 2
                self.K = out_channels // self.U
            if out_channels % competitors != 0:
                raise ValueError('Incorrect number of competitors. '
                                 'Cant divide {} feature maps in groups of {}..'.format(out_channels, competitors))
        else:
            if competitors != 1:
                print('Wrong value of competitors for activation ' + activation + '. Setting value to 1.')
                self.U = 1
                self.K = out_channels // self.U

        self.posterior_mean = Parameter(torch.Tensor(self.out_channels, self.in_channels,
                                                     self.kernel_size, self.kernel_size))

        if not deterministic:
            # posterior unnormalized scale. Needs to be passed by softplus
            self.posterior_un_scale = Parameter(torch.Tensor(self.out_channels, self.in_channels,
                                                         self.kernel_size, self.kernel_size))
            self.register_buffer('weight_eps', None)

        if bias:
            self.bias_mean = Parameter(torch.Tensor(out_channels))
            if not self.deterministic:
                self.bias_un_scale = Parameter(torch.Tensor(out_channels))
                self.register_buffer('bias_eps', None)
        else:
            self.register_parameter('bias_mean', None)
            if not self.deterministic:
                self.register_parameter('bias_un_scale', None)
                self.register_buffer('bias_eps', None)

        if ibp:
            self.conc1 = Parameter(torch.Tensor(self.K))
            self.conc0 = Parameter(torch.Tensor(self.K))
            self.t_pi = Parameter(torch.Tensor(self.K))

        else:
            self.register_parameter('conc1', None)
            self.register_parameter('conc0', None)
            self.register_parameter('t_pi', None)

        if self.ibp:
            self.ibp_params = nn.ParameterList([self.conc1, self.conc0, self.t_pi])

        self.params = nn.ParameterList([self.posterior_mean])

        #if activation=='lwta':
        #    self.params.extend([self.temp])
        if bias:
            self.params.extend([self.bias_mean])
        if not deterministic:
            self.params.append(self.bias_un_scale)
            self.params.append(self.posterior_un_scale)

        self.reset_parameters()




    def reset_parameters(self):
        """
        Initialization function for all the parameters of the Dense Bayesian Layer.

        :return: null
        """

        # original init
        init.kaiming_normal_(self.posterior_mean, mode='fan_out')
        if not self.deterministic:
            self.posterior_un_scale.data.fill_(-5.)

        if self.bias:
            init.constant_(self.bias_mean, 0)

        if self.activation =='lwta':
            init.constant_(self.temp, -0.01)

        if not self.deterministic:
            self.bias_mean.data.fill_(0.0)

        if self.batch_norm:
            init.constant_(self.bn_layer.weight, 1)
            init.constant_(self.bn_layer.bias, 0)

        if self.ibp:
            self.conc1.data.fill_(2.)
            self.conc0.data.fill_(0.5453)

            init.uniform_(self.t_pi, .1, 5.)

    def forward(self, input):
        """
        Overrride the forward pass to match the necessary computations.

        :param input: torch tensor: the input to the layer.

        :return: torch tensor: the output of the layer after activation
        """
        layer_loss = 0.

        if self.training:

            if not self.deterministic:
                # use the reparameterization trick
                posterior_scale = F.softplus(self.posterior_un_scale)
                W = self.posterior_mean + posterior_scale * torch.randn_like(posterior_scale)
                kl_weights = -0.5 * torch.mean(2 * posterior_scale - torch.square(self.posterior_mean)
                                               - posterior_scale ** 2 + 1)
                layer_loss += torch.sum(kl_weights)
            else:
                W = self.posterior_mean

            if self.ibp:
                z, kl_sticks, kl_z = self.indian_buffet_process(self.temperature)

                W = z*W

                layer_loss += kl_sticks
                layer_loss += kl_z


            if self.bias:
                if not self.deterministic:
                    bias = self.bias_mean + F.softplus(self.bias_un_scale) * torch.randn_like(self.bias_un_scale)
                else:
                    bias = self.bias_mean
            else:
                bias = None


        else:
            W = self.posterior_mean
            bias = self.bias_mean

            if self.ibp:
                z, _, _ = self.indian_buffet_process(0.01)
                W = z*W

        out = F.conv2d(input, W, bias, stride = self.stride, padding = self.padding)

        if self.batch_norm:
            out = self.bn_layer(out)

        if self.activation == 'linear':
            self.kl_ = layer_loss
            return out

        elif self.activation == 'relu':
            self.kl_ = layer_loss
            self.acts = out
            return F.relu(out)

        elif self.activation == 'lwta':
            out, kl = self.lwta_activation(out, F.softplus(self.temp) if self.training else 0.01)
            layer_loss += kl
            self.kl_ = layer_loss

            return out

        elif self.activation == 'hard_lwta':
            x = torch.reshape(out, [-1, out.size(1) // self.U, self.U, out.size(-2),out.size(-1)])
            a = torch.argmax(x, 2, keepdims=True)
            mask = torch.zeros_like(x).scatter_(2, a, 1.).reshape(out.shape)

            self.kl_ = layer_loss
            out = out*mask

            return out

        else:
            raise ValueError(self.activation + " is not implemented..")


    def indian_buffet_process(self, temp =0.67):
        """
            Compute the KL divergence for the infian buffet process imposed on the synapses
            of a convolutional layer.
            If we are in eval mode, we utilize the cut-off threshold to remove connections from the architecture.
        """

        kl_sticks = kl_z = 0.
        z_sample = bin_concrete_sample(self.t_pi, temp)

        if not self.training:
            t_pi_sigmoid = torch.sigmoid(self.t_pi)
            mask = t_pi_sigmoid > self.tau

            z_sample = mask*t_pi_sigmoid

        z = z_sample.repeat(self.U)

        # compute the KL divergence during training
        if self.training:
            a_soft = F.softplus(self.conc1)
            b_soft = F.softplus(self.conc0)

            q_u = kumaraswamy_sample(a_soft, b_soft, sample_shape = [a_soft.size(0)])
            prior_pi = torch.cumprod(q_u, -1) + 1e-3

            q = torch.sigmoid(self.t_pi)
            log_q = torch.log(q + 1e-8)
            log_p = torch.log(prior_pi + 1e-8)

            kl_z = (q*(log_q - log_p)).mean()
            kl_sticks = torch.sum(kl_divergence_kumaraswamy(torch.ones_like(a_soft), a_soft, b_soft))


        return z[:, None, None, None], kl_sticks, kl_z



    def lwta_activation(self, input, temp):
        """
        Function implementing the LWTA activation with a competitive random sampling procedure as described in
        Panousis et al., Nonparametric Bayesian Deep Networks with Local Competition.

        :param hard: boolean: flag to draw hard samples from the binary relaxations.
        :param input: torch tensor: the input to compute the LWTA activations.

        :return: torch tensor: the output after the lwta activation
        """

        kl = 0.

        logits = torch.reshape(input, [-1, self.K, self.U, input.size(-2), input.size(-1)])

        if self.training:
            xi = concrete_sample(logits, temp, axis = 2)
        else:
            xi = concrete_sample(logits, 0.01, axis = 2)

        out = logits * xi#.reshape([-1, self.K, self.U, 1,1])


        # set some parameters for plotting
        self.probs = F.softmax(logits.mean([3,4]), 2).reshape([-1, self.K, self.U])
        self.acts = out

        out = torch.reshape(out, input.shape)

        # compute the KL divergence during training
        if self.training:
            q = xi
            log_q = torch.log(q + 1e-8)

            kl = torch.mean(q*(log_q - torch.log(torch.tensor(1.0/ self.U))), 0)
            kl = torch.sum(kl)/1000.

        return out, kl


    def extra_repr(self):
        """
        Print some stuff about the layer parameters.

        :return: str: string of parameters of the layer.
        """

        return " input_features = {}, output_features = {}, bias = {}".format(
            self.in_channels, self.out_channels, self.bias
        )


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

    def __init__(self, inplace = True, deterministic = False, U=2):
        super(LWTA, self).__init__()
        self.inplace = inplace
        self.temperature = -.01
        self.temp_test = 0.01
        self.deterministic = deterministic
        self.kl_ = 0.
        self.U = U
        self.temp = Parameter(torch.tensor(self.temperature))
        #self.params = nn.ParameterList([self.temp])

    def forward(self, input):
        out, kl = lwta_activation(input, U = self.U, training = self.training,
                                  temperature = F.softplus(self.temp),
                                  deterministic=self.deterministic, temp_test = self.temp_test)
        self.kl_ = kl

        return out

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


def lwta_activation(input, U = 2, deterministic = False, training = True, temperature = 0.67, temp_test = 0.01):
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

        if deterministic:
            a = torch.argmax(logits, 2, keepdims = True)
            mask_r = torch.zeros_like(logits).scatter_(2, a, 1.).reshape(input.shape)
        else:
            if training:
                mask = concrete_sample( logits, 0.67)
            else:
                mask = concrete_sample(logits, temp_test  )
            mask_r = mask.reshape(input.shape)
    else:
        x = torch.reshape(out, [-1, out.size(1)//U, U, out.size(-2), out.size(-1)])

        if deterministic:
            a = torch.argmax(x, 2, keepdims = True)
            mask_r = torch.zeros_like(x).scatter_(2, a , 1.).reshape(input.shape)
        else:
            logits = x
            if training:
                mask = concrete_sample(logits, temperature, axis = 2)
            else:
                mask = concrete_sample(logits , temp_test, axis = 2)

            mask_r = mask.reshape(input.shape)

    if False:# training and not deterministic:
        q = mask
        log_q = torch.log(q + 1e-8)
        log_p = torch.log(torch.tensor(1.0 / U))

        kl = torch.sum(q * (log_q - log_p), 1)
        kl = torch.mean(kl) / 1000.

    input *= mask_r

    return input, kl


