# EEGNet-tutorial
This is a naive Pytorch implementation of [EEGNet](https://arxiv.org/abs/1611.08024)
We don't consider data preprocessing, visualization , ablation test, within/cross-subject experiments.

## Setup environment
0. Clone this repo 
```
git clone https://github.com/AilurusUmbra/EEGNet-tutorial.git
cd EEGNet-tutorial
```
1. From conda (**recommended**)
```
conda env create -f environment.yml
```

2. From pip
```
pip install -r requirements.txt
```

## Run
1. Start to train EEGNet
```
python run.py
```
2. Distributed tuning hyperparameters 
* Using [ray tune](https://docs.ray.io/en/latest/tune.html)
> Noted that the path of dataset in `dataloader.py` should be modified to absolute path,
> since that ray tune generates several worker processes to run training function at different working paths.
```
python tuning.py
```

##### [Notes](https://hackmd.io/Z5brl6A6QPGhxrs_5YcunQ?view)
