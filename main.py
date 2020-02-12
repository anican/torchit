from project import Project
import multiprocessing
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.functional as F
import torch.optim as optim
from utils import pt_util
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

    # Data Transforms and Datasets
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    data_train = datasets.CIFAR10(root=str(project.DATA_PATH), train=True, download=True, transform=transform_train)
    data_test = datasets.CIFAR10(root=str(project.DATA_PATH), train=False, download=True, transform=transform_test)

    use_cuda = torch.cuda.is_available()
    # torch.manual_seed(args.seed)
    torch_device = torch.device("cuda" if use_cuda else "cpu")
    print('using device', torch_device)
    print('num cpus:', multiprocessing.cpu_count())
    kwargs = {'num_workers': multiprocessing.cpu_count(), 'pin_memory': True} if use_cuda else {}
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # replace with get_dataloaders() in the template overall
    train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(data_test, batch_size=args.test_batch_size, **kwargs)

    model = GoogLeNet.to(device=torch_device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    start_epoch = model.load_last_model(str(project.DATA_PATH) + 'checkpoints')

    train_losses, test_losses, test_accuracies = pt_util.read_log(LOG_PATH, ([], [], []))

