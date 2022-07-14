from tqdm import tqdm
import pylab
from HierarchicalProbUNet import HierarchicalProbUNet
from load_LIDC_data import LIDC_IDRI
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils import *
from torchvision import transforms

# transform = transforms.Compose([
#     transforms.Normalize(mean=0.5, std=0.5)
# ])
# dataset = LIDC_IDRI(dataset_location='./', transform=transform)

dataset = LIDC_IDRI(dataset_location='./')
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.1 * dataset_size))
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(dataset, batch_size=8, sampler=train_sampler, pin_memory=True)
test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler)
print("Number of training/test patches:", (len(train_indices),len(test_indices)))

net = HierarchicalProbUNet()
net.to('cuda')
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)
epochs = 50
epoch_loss = []


def epoch_train(epoch):
    for x, y, _ in tqdm(train_loader):
        x, y = x.to('cuda'), y.to('cuda')
        y = torch.unsqueeze(y, 1)
        loss_dict = net.sum_loss(y, x, mask=None)
        reg_loss = l2_regularisation(net.prior) + l2_regularisation(net.posterior) + l2_regularisation(net.f_comb.decoder)
        loss = loss_dict['supervised_loss'] + 1e-5 * reg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch:', epoch, 'loss:', round(loss.item(), 3))
    epoch_loss.append(loss.item())
    if loss.item() < 3:
        static_dict = net.state_dict()
        torch.save(static_dict, './weights/epoch_{},loss_{}.pth'.format(epoch, round(loss.item(), 3)))


if __name__ == '__main__':
    # img, label, _ = next(iter(train_loader))
    # torch.Size([5, 1, 128, 128])
    # torch.Size([5, 128, 128])
    # print(img.shape)
    # print(label.shape)

    # plt.figure(figsize=(12, 8))
    # for i, (img, label) in enumerate(zip(img[:4], label[:4])):
    #     img = img.permute(1, 2, 0).numpy()
    #     label = label.numpy()
    #     plt.subplot(2, 4, i + 1)
    #     plt.imshow(img)
    #     plt.subplot(2, 4, i + 5)
    #     plt.imshow(label)
    # pylab.show()

    net.train()
    for epoch in range(epochs):
        epoch_train(epoch)

    plt.plot(range(1, epochs + 1), epoch_loss, label='loss')
    plt.legend()
    plt.savefig('./my_figure.png')
    pylab.show()

