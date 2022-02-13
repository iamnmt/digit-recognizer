import torch
import numpy as np
import pandas as pd

__all__ = ['MNISTDataset']

class MNISTDataset():
    def __init__(self, csv_path, is_rgb=False, is_train=True):
        df = pd.read_csv(csv_path)
        self.is_train = is_train
        if is_train:
            self.data = df.iloc[:, 1:].values.reshape(-1, 1, 28, 28)
            self.labels = df.iloc[:, 0].values
        else:
            self.data = df.values.reshape(-1, 1, 28, 28)
            self.labels = np.arange(1, self.data.shape[0]+1)
            
        if is_rgb:
            self.data = self.data.repeat(3, 1)

    def __getitem__(self, i):
        img = torch.FloatTensor(self.data[i]) / 255.
        lbl = self.labels[i]
        return img, lbl

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    ds = MNISTDataset('data/train.csv')
    print(len(ds))
    for i, (im, lbl) in enumerate(ds):
        print(im.shape, lbl)
        break

    ds = MNISTDataset('data/train.csv', is_rgb=True)
    print(len(ds))
    for i, (im, lbl) in enumerate(ds):
        print(im.shape, lbl)
        break
