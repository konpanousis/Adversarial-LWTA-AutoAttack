from __future__ import print_function
import os
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from models.wideresnet_lwta import *
from attacks.pgd import madry_loss

from torch.utils.tensorboard import SummaryWriter
import time


parser = argparse.ArgumentParser(description='PyTorch CIFAR Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', type=float, default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', type=int, default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', type=float, default=0.007,
                    help='perturb step size')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-cifar-wideResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=5, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--train-nat', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--adv-method', default='pgd',
                    help='inner_max_type ')
parser.add_argument('--width', type=int, default=1,
                    help='network width')
parser.add_argument('--no-early-stop', action='store_true', default=False,
                    help='not use early stop learning rate schedule')
parser.add_argument('--beta', type=float, default=6.0,
                    help='lambda parameter of robust regularization')

args = parser.parse_args()

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    os.makedirs(os.path.join(model_dir,'summary'))
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root='~/data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
testset = torchvision.datasets.CIFAR10(root='~/data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

writer = SummaryWriter(os.path.join(model_dir,'summary'))

def train(args, model, device, train_loader, optimizer, epoch, beta):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        global_step = int((epoch-1) * (50000 / 128) + batch_idx)
        
        if args.adv_method == 'pgd':
            adv_loss_func = madry_loss
        else:
            adv_loss_func = None

        if args.train_nat:
            loss = F.cross_entropy(model(data), target)
        else:
            loss, x_adv = adv_loss_func(
                model=model, x_natural=data, y=target, optimizer=optimizer,
                step_size=args.step_size, epsilon=args.epsilon, perturb_steps=args.num_steps,
                beta=beta)
        
        loss.backward()

        # record gradients' L1 norm
        for name, param in model.named_parameters():
            if 'weight' in name and param.grad is not None:
                summary_values = torch.mean(torch.abs(param.grad.clone()))
                writer.add_scalar(name, summary_values,  global_step=global_step)
        
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=0.031,
                  num_steps=20,
                  step_size=0.007,
                  random=True):
    out = model(X)
    nat_pred = out.data.max(1)[1]
    err = (nat_pred != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    adv_pred = model(X_pgd).data.max(1)[1]
    err_pgd = (adv_pred != y.data).float().sum()

    # smooth erro
    err_smth = (adv_pred != nat_pred).float().sum()

    return err, err_pgd, err_smth


def eval_adv(model, device, test_loader, epoch):
    model.eval()

    robust_err_total = 0
    natural_err_total = 0
    smooth_err_total = 0
    total = 0
    global_step = int(epoch * (50000 / 128))

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust, err_smth = _pgd_whitebox(
                model, X, y, 0.031, 20, 0.003)
        robust_err_total += err_robust
        natural_err_total += err_natural
        smooth_err_total += err_smth
        total += len(data)

    writer.add_scalar(
        'acc_nat', 1 - natural_err_total / total, global_step=global_step)
    writer.add_scalar(
        'acc_adv', 1 - robust_err_total / total, global_step=global_step)
    writer.add_scalar(
        'acc_smth', 1 - smooth_err_total / total, global_step=global_step)

    return (1 - robust_err_total / total)


def get_tailored_optim():
    bn_params = []
    other_params = []
    filter_word = 'bn'
    for name, param in model.named_parameters():
        if filter_word in name:
            bn_params.append(param)
            print(filter_word)
            print(name)
        else:
            other_params.append(param)
            print('other')
            print(name)

    optimizer = optim.SGD([
        {'params': bn_params, 'weight_decay':0},
        {'params': other_params, 'weight_decay':args.weight_decay}
        ], momentum=args.momentum, lr=args.lr)

    return optimizer
   

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    global_step = int((epoch - 1) * (50000 / 128))
    writer.add_scalar('learning_rate', lr, global_step=global_step)


def lr_early_stop(optimizer, epoch):
    """prevent over-fitting"""
    lr = args.lr
    if epoch >=75:
        lr = lr * (0.5 ** (epoch-74))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    global_step = int((epoch - 1) * (50000 / 128))
    writer.add_scalar('learning_rate', lr, global_step=global_step)


def main():

    widths = [10]

    # lambda does not play a role in PGD
    lamb = [6]

    for w in widths:
        for l in lamb:

            model = WideResNet(widen_factor=w).to(device)
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

            if args.adv_method == 'pgd':
                print('=========== PGD ==============')
            else:
                print('Unknown Methods')
            print('===== Width: %d Reg: %d'%(w,l))
            # Resume if ckpt exist
            init_epoch = 1
            cur_model_dir = model_dir +'/' + args.adv_method +'/width_%s_l_%s'%(w,l)
            os.makedirs(cur_model_dir, exist_ok = True)
            ckpt_path = os.path.join(cur_model_dir, 'ckpt.pt')
            highest_path = os.path.join(cur_model_dir, 'highest.pt')
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                init_epoch = checkpoint['epoch']+1

            highest_acc = 0
            for epoch in range(init_epoch, args.epochs + 1):
                start_time = time.time()
                # adjust learning rate for SGD
                if args.no_early_stop:
                    adjust_learning_rate(optimizer, epoch)
                else:
                    lr_early_stop(optimizer, epoch)
                # adversarial training
                train(args, model, device, train_loader, optimizer, epoch, l)
                print('Train time: %s secs'%(time.time()-start_time))

                start_time = time.time()
                # evaluation on natural examples
                acc_robust = eval_adv(model, device, test_loader, epoch)
                state = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                }
                torch.save(state, ckpt_path)
                if acc_robust > highest_acc:
                    print('New Peak %.4f'%acc_robust)
                    highest_acc = acc_robust
                    torch.save(state, highest_path)
                print('Eval time: %s secs'%(time.time()-start_time))
                print('======================================')
                print('======================================')

            del model, optimizer

if __name__ == '__main__':
    main()
