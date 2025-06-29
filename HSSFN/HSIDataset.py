import numpy as np
import torch
from torch.utils.data.dataset import Dataset



class HSIDataset(Dataset):
    def __init__(self, dataset, transfor):
        super(HSIDataset, self).__init__()
        # 将传入的数据集dataset 的第一个元素
        # 转换为 np.float32 类型，并存储在 self.data 中。
        # 这是为了确保数据类型与深度学习模型的要求相匹配

        self.data = dataset[0]
        self.labels = []
        # 遍历数据集 dataset 的第二个元素（通常是标签数据），
        # 将每个标签转换为整数类型后添加到 self.labels 列表中。
        for n in dataset[1]: self.labels += [int(n)]

    def __getitem__(self, index):
        img = torch.from_numpy(np.asarray(self.data[index,:,:,:]).astype(np.float32))
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.labels)

    def __labels__(self):
        return self.labels


