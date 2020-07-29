from __future__ import print_function

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from dataloader import read_bci_data
from EEGNet import (EEGNet, train, test)

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

activation_list = [nn.ELU, nn.ReLU, nn.ReLU6, nn.LeakyReLU]


def gen_bci_dataloaders():
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

def get_data_loaders():
    train_dataset, test_dataset = gen_bci_dataloaders()
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
    return train_loader, test_loader

# Change these values if you want the training to run quicker or slower.
EPOCH_SIZE = 512
TEST_SIZE = 256

# Training settings
parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
parser.add_argument(    "--use-gpu",    action="store_true",    default=True,    help="enables CUDA training")
parser.add_argument(
    "--ray-address", type=str, help="The Redis address of the cluster.")
parser.add_argument(
    "--smoke-test", action="store_true", help="Finish quickly for testing")


class TrainEEG(tune.Trainable):
    def _setup(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader, self.test_loader = get_data_loaders()
        self.model = EEGNet(
                activation=activation_list[config.get("activation", 1)]).to(self.device)
                #dropout1=config.get("drop1", 0.6),
                #dropout2=config.get("drop2", 0.65)
                #).to(self.device)

        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config.get("lr", 0.0025))

    def _train(self):
        train(
            self.model, self.optimizer, self.train_loader, device=self.device)
        acc = test(self.model, self.test_loader, self.device)
        return {"mean_accuracy": acc}

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))



if __name__ == "__main__":
    args = parser.parse_args()
    ray.init(address=args.ray_address, num_cpus=3 if args.smoke_test else None)
    #sched = ASHAScheduler(metric="mean_accuracy")


    analysis = tune.run(
        TrainEEG,
        #scheduler=sched,

        stop={
            "mean_accuracy": 0.87,
            "training_iteration": 3 if args.smoke_test else 300,
        },
        resources_per_trial={
            "cpu": 3,
            "gpu": int(args.use_gpu)
        },
        num_samples=1 if args.smoke_test else 10,
        checkpoint_at_end=True,
        checkpoint_freq=10,
        config={
            "activation": tune.grid_search([0, 1, 2, 3]),
            #"args": args,
            #"drop1": tune.grid_search([0.6,0.65]),
            #"drop2": tune.grid_search([0.65,0.68]),
            "lr": tune.grid_search([0.0025,0.0026])
            #"lr": tune.grid_search([0.0022,0.0025, 0.003, 0.01])
            #"lr": tune.uniform(0.0022, 0.005)
        })

    print("Best config is:", analysis.get_best_config(metric="mean_accuracy"))
