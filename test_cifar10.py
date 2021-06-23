from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from models.wideresnet_lwta import *
from models.resnet import *
from autoattack import AutoAttack


parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', type=int, default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', type=float, default=0.003,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--model-path',
                    default='./model-cifar-wideResNet/highest.pt',
                    help='model for white-box attack evaluation')
parser.add_argument('--source-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='source model for black-box attack evaluation')
parser.add_argument('--target-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='target model for black-box attack evaluation')
parser.add_argument('--black-attack', action='store_true', default=False,
                    help='whether perform white-box attack')
parser.add_argument('--autoattack', action='store_true', default=True,
                    help='whether perform white-box attack')
parser.add_argument('--width', type=int, default=1,
                    help='width of networks')
parser.add_argument('--attack-loss', default='ce',
                    help='inner_max_type')

args = parser.parse_args()


# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])
testset = torchvision.datasets.CIFAR10(root='~/data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def cw_loss(logits, y, confidence=0):
    onehot_y = torch.nn.functional.one_hot(y, num_classes=logits.shape[1]).float()
    self_loss = F.nll_loss(-logits, y, reduction='none')
    other_loss = torch.max((1 - onehot_y) * logits, dim=1)[0]
    return -torch.mean(torch.clamp(self_loss - other_loss + confidence, 0))


def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            if args.attack_loss == 'ce':
                loss = nn.CrossEntropyLoss()(model(X_pgd), y)
            else:
                loss = cw_loss(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    print('err pgd (white-box): ', err_pgd)
    return err, err_pgd


def _pgd_blackbox(model_target,
                  model_source,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out = model_target(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model_source(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    err_pgd = (model_target(X_pgd).data.max(1)[1] != y.data).float().sum()
    print('err pgd black-box: ', err_pgd)
    return err, err_pgd


def eval_adv_test_whitebox(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_err_total: ', natural_err_total)
    print('robust_err_total: ', robust_err_total)


def eval_adv_test_blackbox(model_target, model_source, device, test_loader):
    """
    evaluate model by black-box attack
    """
    model_target.eval()
    model_source.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_blackbox(model_target, model_source, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_err_total: ', natural_err_total)
    print('robust_err_total: ', robust_err_total)


def main():

    if not args.black_attack and not args.autoattack:
        # white-box attack
        print('pgd white-box attack')
        model = WideResNet(widen_factor=args.width).to(device)
        model.load_state_dict(torch.load(args.model_path)['model_state_dict'])

        eval_adv_test_whitebox(model, device, test_loader)
    elif not args.black_attack and args.autoattack:

        widths = [5]  # 10]
        lamb = [6]#, 9, 12, 15, 18, 21]

        for w in widths:
            for l in lamb:
                path = 'model-cifar-wideResNet/pgd/width_%s_l_%s/highest.pt'%(w, l)

                model = WideResNet(widen_factor=w).to(device)
                model.load_state_dict(torch.load(path)['model_state_dict'])
                model.eval()

                adversary = AutoAttack(model, norm='Linf', eps=8./255, version='standard')

                l = [x for (x, y) in test_loader]
                x_test = torch.cat(l, 0)
                l = [y for (x, y) in test_loader]
                y_test = torch.cat(l, 0)
                with torch.no_grad():
                    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=128)

    else:
        # black-box attack
        print('pgd black-box attack')
        model_target = WideResNet(widen_factor=args.width).to(device)
        model_target.load_state_dict(torch.load(args.target_path)['model_state_dict'])
        model_source = WideResNet(widen_factor=args.width).to(device)
        model_source.load_state_dict(torch.load(args.source_path)['model_state_dict'])

        eval_adv_test_blackbox(model_target, model_source, device, test_loader)


if __name__ == '__main__':
    main()
