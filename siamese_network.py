from collections import defaultdict
import random

import numpy as np

from torch import IntTensor

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

from torch.utils.data.sampler import Sampler
from torch.utils.data.dataloader import DataLoader


class SiameseCNN(nn.Module):
    def __init__(self, input_shape=(105,105,1)):
        super(SiameseCNN,self).__init__()
        self.conv1 = nn.Conv2d(1,64,kernel_size=105)
        self.conv2 = nn.Conv2d(64,128,kernel_size=7)
        self.conv3 = nn.Conv2d(128,256,kernel_size=4)
        size = self._get_conv_output(input_shape)
        self.linear = nn.Linear(size,4096)

    def _get_conv_output(self, shape):
        input = Variable(torch.rand(1, *shape))
        h1 = self.conv1(input)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        return h3.view(1,-1).size(1)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        return F.sigmoid(x.linear(x.view(-1)))


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork,self).__init__()
        self.cnn = SiameseCNN()
        self.linear = nn.Linear(4096, 1)

    def forward(self, x1, x2):
        x1 = self.cnn(x1)
        x2 = self.cnn(x2)
        merged = torch.abs(x1 - x2)
        return F.sigmoid(self.linear(merged))

# def create_one_shot_task(samples_map, N):
#     categories = np.random.choice(len(samples_map), N, replace=False)
#     true_category = categories[0]
#
#
#
#
#
# def test_oneshot(model, N, k, val_set):
#     n_correct = 0
#     for i in range(k):
#         inputs, targets = create_one_shot_task(val_set, N)
#         probs = model(inputs)
#         if np.argmax(probs) == 0:
#             n_correct += 1
#     percent_correct = (100*n_correct / k)
#     print('Percent Correct', percent_correct)



class PairSampler(Sampler):
    k_examples = 20
    def __init__(self, dataset):
        super(PairSampler).__init__()
        self.samples_map = defaultdict(list)
        for image, label in dataset:
            self.samples_map[label].append(image)
        self.n_classes = len(self.samples_map)
        self.batch_size = 8

    def __iter__(self):
        return self

    def next(self):
        categories = np.random.choice(self.n_classes, size=self.batch_size, replace=False)
        images_1, images_2 = [], []
        targets = np.zeros(self.batch_size)
        targets[self.batch_size // 2:] = 1
        for i in range(self.batch_size):
            category = categories[i]
            idx1 = np.random.randint(0, self.k_examples)
            image_1 = self.samples_map[category][idx1]
            idx2 = np.random.randint(0, self.k_examples)
            category_2 = category if i >= self.batch_size // 2 else (category + np.random.randint(1,
                                                                                                  self.n_classes)) % self.n_classes
            image_2 = self.samples_map[category_2][idx2]
            images_1.append(image_1)
            images_2.append(image_2)
        yield torch.cat(images_1), torch.cat(images_2), IntTensor(targets)

    def __len__(self):
        return self.batch

omni_dataset = datasets.Omniglot('data',
                transform=transforms.Compose([
                transforms.ToTensor()]))

loader = DataLoader(omni_dataset, sampler=PairSampler(omni_dataset))
for batch in loader:
    import ipdb; ipdb.set_trace()

