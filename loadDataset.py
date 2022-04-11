import torch
import torchvision
import torchvision.transforms as transforms


def load_CIFAR10():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    dataiter = iter(trainloader)
    train_images, train_labels = dataiter.next()

    dataiter = iter(testloader)
    test_images, test_labels = dataiter.next()

    #return train_images, train_labels, test_images, test_labels
    return trainloader, testloader
