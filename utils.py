import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

def get_dataloader(args):
    if args.dataset=="c10":
        args.size = 32
        args.img_c = 3
        args.mean = [0.4914, 0.4822, 0.4465]
        args.std = [0.2470, 0.2435, 0.2616]
    elif args.dataset=="c100":
        args.size = 32
        args.img_c = 3
        args.mean = [0.5071, 0.4867, 0.4408]
        args.std = [0.2675, 0.2565, 0.2761]
    else:
        raise ValueError(f"{args.dataset}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ])
    if args.dataset=="c10":
        ds = torchvision.datasets.CIFAR10("data", train=True, transform=transform, download=True)
    elif args.dataset=="c100":
        ds = torchvision.datasets.CIFAR100("data", train=True, transform=transform, download=True)
    else:
        raise ValueError(f"{args.dataset}")

    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return dl


def get_model(args):
    if args.model_name == "transgan":
        from transgan import Generator, Discriminator
        g = Generator(img_size=args.size, img_c=args.img_c, model_size=args.model_size)
        d = Discriminator(img_size=args.size, img_c=args.img_c, patch=args.patch)
    else:
        raise ValueError(f"{args.model_name}")
    return g, d


def get_criterion(args):
    if args.criterion == "mse":
        criterion = nn.MSELoss()
    elif args.criterion=="ns":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"{args.criterion}?")

    return criterion

def get_optimizer(g, d, args):
    if args.optimizer == "adam":
        g_optimizer = torch.optim.Adam(g.parameters(), lr=args.g_lr, betas=(args.beta1, args.beta2))
        d_optimizer = torch.optim.Adam(d.parameters(), lr=args.d_lr, betas=(args.beta1, args.beta2))
    else:
        raise ValueError(f"{args.optimizer}?")

    return g_optimizer, d_optimizer


def get_experiment_name(args):
    experiment_name = f"{args.model_name}_{args.model_size}"
    print(f"EXPERIMENT: {experiment_name}")
    return experiment_name