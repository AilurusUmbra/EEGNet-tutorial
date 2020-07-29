from functools import reduce

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim
import pandas as pd
from torch.cuda import device
from torch.utils.data import TensorDataset, DataLoader

from dataloader import read_bci_data
from EEGNet import EEGNet
from DCN2 import DeepConvNet

def get_bci_dataloaders():
    train_x, train_y, test_x, test_y = read_bci_data()
    datasets = []
    for train, test in [(train_x, train_y), (test_x, test_y)]:
        train = torch.stack(
            [torch.Tensor(train[i]) for i in range(train.shape[0])]
        )
        test = torch.stack(
            [torch.Tensor(test[i:i+1]) for i in range(test.shape[0])]
        )
        datasets += [TensorDataset(train, test)]

    return datasets

def get_data_loaders(train_dataset, test_dataset):
    #train_dataset, test_dataset = get_bci_dataloaders()
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
    return train_loader, test_loader


def showResult(title='', **kwargs):
    plt.figure(figsize=(15, 7))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')

    for label, data in kwargs.items():
        plt.plot(range(len(data)), data, '--' if 'test' in label else '-', label=label)
    plt.ylim(0, 100)
    plt.xlim(0, 300)
    points = [(-5, 87), (310, 87)]
    (xpoints, ypoints) = zip(*points)

    plt.plot(xpoints, ypoints, linestyle='--', color='black')

    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.show()


    
def main():
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nets1 = {
        "EEG_elu": EEGNet(nn.ELU).to(device),
        "EEG_relu": EEGNet(nn.ReLU).to(device),
        "EEG_relu6": EEGNet(nn.ReLU6).to(device),
        "EEG_leaky_relu": EEGNet(nn.LeakyReLU).to(device)
    }

    nets2 = {
            "DCN_elu": DeepConvNet(nn.ELU).to(device),
            "DCN_relu": DeepConvNet(nn.ReLU).to(device),
            "DCN_relu6": DeepConvNet(nn.ReLU6).to(device),
            "DCN_leaky_relu": DeepConvNet(nn.LeakyReLU).to(device)
    }
    
    nets = nets1
    #nets = nets2
    
    # Training setting
    loss_fn = nn.CrossEntropyLoss()
    learning_rates = {0.0025}

    optimizer = torch.optim.Adam
    optimizers = {
        key: optimizer(value.parameters(), lr=learning_rate, weight_decay=0.0001)
        for key, value in nets.items()
        for learning_rate in learning_rates
    }

    epoch_size = 300
    batch_size = 64
    acc = train(nets, epoch_size, batch_size, loss_fn, optimizers)
    df = pd.DataFrame.from_dict(acc)
    
    #df.to_csv('eeg_0025_0001.csv')
    print(df)
    #display(df)
    return df

# This train is for demo and recording accuracy
def train(nets, epoch_size, batch_size, loss_fn, optimizers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainDataset, testDataset = get_bci_dataloaders()
    trainLoader, testLoader = get_data_loaders(trainDataset, testDataset)
    
    accuracy = {
        **{key + "_train": [] for key in nets},
        **{key + "_test": [] for key in nets}
    }
    for epoch in range(epoch_size + 1):
        train_correct = {key: 0.0 for key in nets}
        test_correct = {key: 0.0 for key in nets}
        for step, (x, y) in enumerate(trainLoader):
            x = x.to(device)
            y = y.to(device).long().view(-1)

            for key, net in nets.items():
                net.train(mode=True)
                y_hat = net(x)
                loss = loss_fn(y_hat, y)
                loss.backward()
                train_correct[key] += (torch.max(y_hat, 1)[1] == y).sum().item()

            for optimizer in optimizers.values():
                optimizer.step()
                optimizer.zero_grad()

        with torch.no_grad():
            for step, (x, y) in enumerate(testLoader):
                x = x.to(device)
                y = y.to(device).long().view(-1)
                for key, net in nets.items():
                    net.eval()
                    y_hat = net(x)
                    test_correct[key] += (torch.max(y_hat, 1)[1] == y).sum().item()

        for key, value in train_correct.items():
            accuracy[key + "_train"] += [(value * 100.0) / len(trainDataset)]

        for key, value in test_correct.items():
            accuracy[key + "_test"] += [(value * 100.0) / len(testDataset)]

        if epoch % 100 == 0:
            print('epoch : ', epoch, ' loss : ', loss.item())
            print(pd.DataFrame.from_dict(accuracy).iloc[[epoch]])
            #display(pd.DataFrame.from_dict(accuracy).iloc[[epoch]])
            print('')
        torch.cuda.empty_cache()
    showResult(title='Activation function comparison(EEGNet)'.format(epoch + 1), **accuracy)
    #showResult(title='Activation function comparison(DCN)'.format(epoch + 1), **accuracy)
    return accuracy

if __name__ == '__main__':
    df1 = main()
    df1.max()
    print(pd.DataFrame(df1.max(), columns=['best_acc']).T)
    #display(pd.DataFrame(df1.max(), columns=['best_acc']).T)
