from project import Project
import multiprocessing
import time
import torch
from torch import nn
import torch.functional as F
import torch.optim as optim
import argparse
from models import GoogLeNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='<Add Project Description>')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--print_interval', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    project = Project()
    # torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('using device', device)
    print('num cpus:', multiprocessing.cpu_count())
    kwargs = {'num_workers': multiprocessing.cpu_count(), 'pin_memory': True} if use_cuda else {}
    # class_names = [<insert names>*], ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # train_loader, val_loader, test_loader = get_dataloaders(
    #     project.DATA_PATH / 'train',
    #     project.DATA_PATH / 'val',
    #     kwargs
    # )
    model = GoogLeNet.to(device=device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    start_epoch = model.load_last_model(DATA_PATH + 'checkpoints')

