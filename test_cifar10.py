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
from autoattack import AutoAttack


parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--autoattack', action='store_true', default=True,
                    help='whether perform white-box attack')
parser.add_argument('--width', type=int, default=1,
                    help='width of networks')

args = parser.parse_args()


# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])
testset = torchvision.datasets.CIFAR10(root='~/data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def main():

    if args.autoattack:

        w=1
        l = 6
        
        path = 'model-cifar-wideResNet/ckpt.pt'

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


if __name__ == '__main__':
    main()
