import numpy as np
import torch
from torch.utils.data import Dataset

root = str(np.load('C:/root/root.npy'))


class CustomDataset(Dataset):
    def __init__(self, data):
        self.x_data = data.tolist()

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        return x


class CustomDataset_label(Dataset):
    def __init__(self, data, label):
        self.x_data = data.tolist()
        self.label_data = label.tolist()

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        labels = torch.FloatTensor(self.label_data[idx])
        return x, labels


def dataset_valid(valid_d, valid_l, batch_size, small=0):
    if small == 0:
        valid_data = np.load(valid_d)  # [:10,:,:,:]
        datalabel = np.load(valid_l)
    else:
        valid_data = np.load(valid_d)[:small, :, :, :]
        datalabel = np.load(valid_l)[:small, :]
    valid_shape = valid_data.shape
    print('data shape = ', valid_shape)
    print("num nan = ", str(np.sum(np.isnan(valid_data))))
    print('label shape = ', datalabel.shape)
    dataset = CustomDataset_label(valid_data, datalabel)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    del valid_data, dataset

    return valid_loader, valid_shape


def dataset_single(valid_d, batch_size, small=0):
    if small == 0:
        valid_data = np.load(valid_d)  # [:10,:,:,:]
    else:
        valid_data = np.load(valid_d)[:small, :, :, :]
    valid_shape = valid_data.shape
    print('data shape = ', valid_shape)
    print("num nan = ", str(np.sum(np.isnan(valid_data))))
    dataset = CustomDataset(valid_data)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    del valid_data, dataset

    return valid_loader, valid_shape


def dataset(train_d, train_l, batch_size, small=0, shuffle=True):
    if small == 0:
        train_data = np.load(train_d)
    else:
        train_data = np.load(train_d)[:small, :, :]

    train_shape = train_data.shape
    #print('data shape = ', train_shape)
    #print("num nan = ", str(np.sum(np.isnan(train_data))))

    datalabel = np.load(train_l)
    #print('label shape = ', datalabel.shape)

    dataset = CustomDataset(train_data)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                               drop_last=True)  # num_workers=2코어수
    del train_data, dataset

    return train_loader, train_shape, datalabel
