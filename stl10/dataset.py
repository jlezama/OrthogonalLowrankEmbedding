import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from IPython import embed
import os
from torch.utils.data.sampler import SubsetRandomSampler

def get(batch_size, data_root='./data/', train=True, val=True, num_samples=500, validation=0, data_augment=1, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'stl10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building STL10 data loader with %i workers, %i samples per class, data augmentation=%i, validation=%i" % (num_workers, num_samples, data_augment, validation))
    ds = []

    
    transform_train = transforms.Compose([
                      transforms.Pad(4),
                      transforms.RandomCrop(96),
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

    transform_test =  transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

    if data_augment==0:
        print 'no data augmentation'
        transform_train = transform_test
        
        
    train_set = datasets.STL10(
                root=data_root, split='train', download=True,
                transform=transform_train)

    test_set = datasets.STL10(
                root=data_root, split='test', download=True,
                transform=transform_test)

    if validation:
        num_samples = 450
    
    
    # select random subset for validation
    N = len(train_set)
    train_size = int(N*float(num_samples)/500.)
    valid_size = N-train_size
    
    print 'number of training examples is %i/%i' % (train_size,N)
    indices = torch.randperm(N)
    train_indices = indices[:train_size]

    if train:
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size, sampler=SubsetRandomSampler(train_indices), **kwargs)
        ds.append(train_loader)
    
    if val:
        if validation:
            valid_indices = indices[len(indices)-valid_size:]
            test_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size, sampler=SubsetRandomSampler(valid_indices), **kwargs)
        else:
            test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)

    ds = ds[0] if len(ds) == 1 else ds
    return ds

if __name__ == '__main__':
    train_ds, test_ds = get(200, num_workers=1)
    for data, target in train_ds:
        print("~~")
